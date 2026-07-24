/**
 * Causal Systems digest subscriptions — Cloudflare Worker.
 *
 * Routes:
 *   POST /subscribe        {email}            -> stores pending, sends double-opt-in mail
 *   GET  /confirm?t=TOKEN                     -> confirms subscription
 *   GET  /unsubscribe?t=TOKEN                 -> removes subscription
 *   POST /send             Bearer SEND_KEY    -> sends body text to all confirmed subscribers
 *
 * Bindings (wrangler.toml): SUBS (KV namespace)
 * Secrets: RESEND_API_KEY, SEND_KEY
 * Vars: FROM_ADDR, BASE_URL, ALLOWED_ORIGIN
 */

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

function cors(env) {
  return {
    "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN || "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

function json(data, status, env) {
  return new Response(JSON.stringify(data), {
    status: status || 200,
    headers: { "Content-Type": "application/json", ...cors(env) },
  });
}

function page(title, body) {
  const html = `<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>${title} · Causal Systems</title>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;800&family=Space+Mono:wght@700&display=swap" rel="stylesheet">
<body style="margin:0;background:#F4F1EA;color:#1F2937;font-family:Archivo,system-ui,sans-serif;display:flex;min-height:100vh;align-items:center;justify-content:center">
<div style="max-width:460px;padding:40px 28px;text-align:center">
<div style="font-family:'Space Mono',monospace;font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:#2B53FF;margin-bottom:14px">Causal Systems · EUA regime digest</div>
<h1 style="font-weight:800;font-size:28px;letter-spacing:-.02em;color:#141414;margin:0 0 12px">${title}</h1>
<p style="font-size:15.5px;line-height:1.6;margin:0">${body}</p>
</div></body>`;
  return new Response(html, { headers: { "Content-Type": "text/html; charset=utf-8" } });
}

async function sendMail(env, to, subject, text) {
  const r = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ from: env.FROM_ADDR, to: [to], subject, text }),
  });
  if (!r.ok) throw new Error(`resend ${r.status}: ${await r.text()}`);
}

async function rateLimited(env, ip) {
  const key = `rl:${ip}`;
  const n = parseInt((await env.SUBS.get(key)) || "0", 10);
  if (n >= 5) return true;
  await env.SUBS.put(key, String(n + 1), { expirationTtl: 3600 });
  return false;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors(env) });
    }

    // ---- subscribe ----
    if (url.pathname === "/subscribe" && request.method === "POST") {
      let email;
      try {
        email = ((await request.json()).email || "").trim().toLowerCase();
      } catch {
        return json({ ok: false, error: "bad request" }, 400, env);
      }
      if (!EMAIL_RE.test(email) || email.length > 254) {
        return json({ ok: false, error: "invalid email" }, 400, env);
      }
      const ip = request.headers.get("CF-Connecting-IP") || "unknown";
      if (await rateLimited(env, ip)) {
        return json({ ok: false, error: "too many requests" }, 429, env);
      }
      if (await env.SUBS.get(`sub:${email}`)) {
        return json({ ok: true, note: "already subscribed" }, 200, env);
      }
      const token = crypto.randomUUID();
      await env.SUBS.put(`pending:${token}`, email, { expirationTtl: 172800 });
      await sendMail(
        env,
        email,
        "Confirm your subscription · Causal Systems EUA digest",
        `You (or someone using this address) asked to receive the EUA regime digest:\n` +
          `a short email only when the regime flips or TNAC crosses an MSR threshold.\n\n` +
          `Confirm the subscription:\n${env.BASE_URL}/confirm?t=${token}\n\n` +
          `If this was not you, ignore this email and nothing will be stored.\n\n` +
          `Causal Systems, a brand of Yarim Trade UG · causalsystems.co`
      );
      return json({ ok: true }, 200, env);
    }

    // ---- confirm ----
    if (url.pathname === "/confirm" && request.method === "GET") {
      const t = url.searchParams.get("t") || "";
      const email = await env.SUBS.get(`pending:${t}`);
      if (!email) return page("Link expired", "This confirmation link is invalid or has expired. Subscribe again on the monitor page.");
      const utoken = crypto.randomUUID();
      await env.SUBS.put(`sub:${email}`, utoken);
      await env.SUBS.put(`unsub:${utoken}`, email);
      await env.SUBS.delete(`pending:${t}`);
      return page("Subscription confirmed", "You will receive the digest only when the regime state changes. Every email contains an unsubscribe link.");
    }

    // ---- unsubscribe ----
    if (url.pathname === "/unsubscribe" && request.method === "GET") {
      const t = url.searchParams.get("t") || "";
      const email = await env.SUBS.get(`unsub:${t}`);
      if (!email) return page("Already unsubscribed", "This unsubscribe link is invalid or was already used.");
      await env.SUBS.delete(`sub:${email}`);
      await env.SUBS.delete(`unsub:${t}`);
      return page("Unsubscribed", "Your address has been removed. No data about you remains stored.");
    }

    // ---- send (called by the GitHub workflow) ----
    if (url.pathname === "/send" && request.method === "POST") {
      const auth = request.headers.get("Authorization") || "";
      if (auth !== `Bearer ${env.SEND_KEY}`) {
        return json({ ok: false, error: "unauthorized" }, 401, env);
      }
      const text = await request.text();
      if (!text.trim()) return json({ ok: false, error: "empty body" }, 400, env);

      let sent = 0, failed = 0, cursor;
      do {
        const list = await env.SUBS.list({ prefix: "sub:", cursor });
        for (const k of list.keys) {
          const email = k.name.slice(4);
          const utoken = await env.SUBS.get(k.name);
          try {
            await sendMail(
              env,
              email,
              "EUA regime digest · state change",
              `${text}\n\n--\nUnsubscribe: ${env.BASE_URL}/unsubscribe?t=${utoken}\n` +
                `Causal Systems, a brand of Yarim Trade UG · causalsystems.co`
            );
            sent++;
          } catch (e) {
            failed++;
          }
        }
        cursor = list.list_complete ? undefined : list.cursor;
      } while (cursor);
      return json({ ok: true, sent, failed }, 200, env);
    }

    return json({ ok: false, error: "not found" }, 404, env);
  },
};
