# Digest subscriptions · setup

One-time setup, roughly 30 minutes. Order matters.

## 1. Resend (the email sender)

1. Create an account at resend.com (free tier: 100 emails/day).
2. Domains -> Add domain -> `causalsystems.co`.
3. Resend shows DNS records (SPF + DKIM). Add them in the Cloudflare
   dashboard under causalsystems.co -> DNS. Wait for "Verified".
4. API Keys -> Create -> copy the key (starts with `re_`).

## 2. Cloudflare Worker

From this folder (`subscribe_worker/`), with Node installed:

    npx wrangler login                      # opens browser, authorize
    npx wrangler kv namespace create SUBS   # prints an id
    # paste the id into wrangler.toml (kv_namespaces)
    npx wrangler deploy                     # prints the workers.dev URL
    # paste that URL into wrangler.toml as BASE_URL, then deploy again:
    npx wrangler deploy

    npx wrangler secret put RESEND_API_KEY  # paste the re_... key
    npx wrangler secret put SEND_KEY        # invent a long random string, e.g.:
                                            #   openssl rand -hex 24
                                            # keep it, the workflow needs it too

## 3. Point the monitor at the Worker

In `eua_monitor/config.py`, set:

    SUBSCRIBE_URL = "https://cs-digest.YOUR-SUBDOMAIN.workers.dev"

(the same URL as BASE_URL). Commit and push; the next build renders a
real subscribe form instead of the mailto link.

## 4. GitHub secrets (repo Settings -> Secrets -> Actions)

    DIGEST_SEND_URL   = https://cs-digest.YOUR-SUBDOMAIN.workers.dev
    DIGEST_SEND_KEY   = the SEND_KEY string from step 2

The workflow posts `site/digest.md` to the Worker whenever the regime
state changes; the Worker mails every confirmed subscriber with a
personal unsubscribe link.

## 5. Test

    curl -X POST https://cs-digest.YOUR-SUBDOMAIN.workers.dev/subscribe \
      -H 'Content-Type: application/json' \
      -d '{"email":"you@example.org"}'

Confirm via the email link, then:

    curl -X POST https://cs-digest.YOUR-SUBDOMAIN.workers.dev/send \
      -H "Authorization: Bearer YOUR_SEND_KEY" \
      -H 'Content-Type: text/plain' \
      --data 'Test digest body.'

Expected: `{"ok":true,"sent":1,"failed":0}` and a mail in your inbox.

## Privacy notes

Double opt-in (nothing stored without confirmation), stored data is
only the address plus an unsubscribe token in Cloudflare KV, every
email carries an unsubscribe link, and unsubscribing deletes the data.
The site's Datenschutz page describes this processing; keep it in sync
if you change the flow.
