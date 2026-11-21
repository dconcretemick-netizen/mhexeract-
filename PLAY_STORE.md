# Google Play Store — MHexeract (project codename 6D)

Everything prepared for a Play Store listing, and the steps only Mick can do
(Google requires the developer himself for payment, identity, and declarations).

## Already done in the code

- **Two build channels** (Gradle product flavors in `android/app/build.gradle.kts`):
  - `direct` — the existing GitHub/sideload build; it keeps updating itself from
    GitHub releases exactly as today.
  - `play` — the Google Play build. Play policy does not allow apps to install
    their own updates, so this flavor:
    - removes `REQUEST_INSTALL_PACKAGES` and `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS`
      from the manifest (`android/app/src/play/AndroidManifest.xml`),
    - carries `BuildConfig.CHANNEL = "play"`, which the gateway reads at startup:
      the Update tab then explains that updates come from Google Play, and the
      self-update endpoints refuse to run.
- **CI builds both**: every push to `main` produces the usual direct APK (release
  asset, unchanged names) **and** a Play bundle artifact named
  `MHexeract-play-aab` (an `.aab` file — Play accepts bundles, not raw APKs).
- The app already targets **API 35**, which meets Play's current target-API
  requirement. The applicationId is `com.mhexeract.app` (renamed before the
  first listing — Play ties a listing to its applicationId permanently, so
  this is the last free chance to change it).

Signing note: Play re-signs apps with Google's own key (Play App Signing). Our
existing CI key becomes the *upload* key. Because the final signature differs,
a phone must stick to one channel — a Play install can't be updated by the
direct APK or vice versa without uninstalling first. That's normal.

## Mick's steps (only he can do these)

1. **Create a Google Play developer account** at play.google.com/console —
   $25 one-time fee, paid with his own card, plus Google's identity
   verification (government ID). Choose a **personal** account unless a
   business identity is wanted.
2. **Create the app** in the Console: name `MHexeract`, default language
   English (US), App (not Game), Free.
3. **Closed test first (Google's rule for new personal accounts):** before a
   production listing, the app must run a **closed test with at least 12
   testers for 14 continuous days**. Upload the AAB (step 5) to the Closed
   testing track, add testers' Gmail addresses (friends/family — they just
   install it and keep it installed), wait out the 14 days, then apply for
   production access in the Console.
4. **Store listing**: paste the copy below. Add at least 2 phone screenshots
   (capture from the phone: Home, Chat, Car diagnostics) and a 512×512 icon
   PNG (the M mark from `icon.svg` exports fine).
5. **Upload the bundle**: download the `MHexeract-play-aab` artifact from the
   latest GitHub Actions run and upload it to the closed-testing release.
   Enroll in **Play App Signing** when prompted (default, recommended).
6. **Declarations** (Console questionnaires; honest answers drafted below):
   privacy policy URL, data safety, content rating, foreground-service
   declaration, ads = none, target audience = 18+ is *not* required — choose
   13+ (the app has a web browser and AI chat, so do not claim it is
   child-directed; it is not in the Families program).
7. **Privacy policy hosting**: the policy text below must live at a public
   URL. Easiest: enable GitHub Pages for this repo and publish it as
   `privacy.html`, or paste it into a GitHub gist and use that link.

## Listing copy

**Title (30 max):** `MHexeract — Free AI Gateway`

**Short description (80 max):**
`Many free AI models in one app, plus car diagnostics, news and games.`

**Full description:**

> MHexeract gives you real AI at no cost — no subscription. Instead of one paid
> plan, it spreads your questions across many free AI providers, plus models
> running on your own devices, and automatically fails over when one hits its
> limit.
>
> • Chat with many AI models through one gateway — MHexeract picks the best
>   available model for each message, or lets several models confer (Council
>   mode) on hard questions
> • Truly anonymous options: keyless, no-account AI models are built in, and
>   free provider keys are kept healthy automatically
> • Private by design: add a local model on your home network and sensitive
>   questions never leave your house. Keys are stored only on your device
> • Car diagnostics: connect a Bluetooth or Wi-Fi OBD-II adapter for live
>   gauges, trouble codes with plain-English descriptions, freeze-frame data,
>   code clearing, and a full emissions-readiness check
> • News from across the political spectrum, a built-in web reader, and a
>   games shelf (Oregon Trail, Solitaire, Tetris, Blackjack and more)
>
> Free tiers have quotas and the best free models trail the top paid ones on
> the hardest reasoning. MHexeract's answer is breadth and failover — many
> providers, a local model as the floor, and background care that keeps the
> pool healthy on its own.

## Privacy policy (draft — publish at a public URL)

> **MHexeract privacy policy**
>
> MHexeract is an AI gateway that runs on your device. The developer collects
> no personal data: there are no accounts, no analytics, no advertising, and
> no tracking.
>
> - **AI keys** you enter are stored only on your device and are sent only to
>   the provider they belong to, to answer your requests.
> - **Messages you send** are transmitted to the AI providers you choose or
>   configure (and to models on your own network, if you add them). Those
>   providers' own privacy policies apply to what you send them. If you use
>   the built-in keyless providers, your messages go to those services.
> - **Car diagnostics** talk over Bluetooth or your local Wi-Fi to your own
>   OBD-II adapter. Vehicle data stays on your device.
> - **News and web features** fetch pages from the sites you open or the
>   bundled news sources; those sites see your IP address as with any browser.
> - Chat history, settings, and game progress are stored only on your device
>   and are removed when you uninstall the app.
>
> Questions: open an issue at https://github.com/dconcretemick-netizen/MCUBED

## Data safety section (draft answers)

- **Data collected by the developer:** None.
- **Data shared with third parties:** The app transmits the messages and
  requests you make to the AI providers you configure — that is the app's
  core function, at your direction, under your own keys/accounts. Declare:
  "App activity / other user-generated content — shared, for app
  functionality." Nothing is sold; no data is used for advertising.
- **Data stored on device:** keys, chat history, settings (not collected).
- **Encryption in transit:** Yes (HTTPS/WSS to providers).
- **Account deletion:** No accounts exist.

## Content rating (expect Teen)

Answer the IARC questionnaire honestly: the app includes user-to-user-free
AI chat (AI-generated content), unrestricted web access via the built-in
browser, and no violence/gambling/sexual content in the app itself. Play's
questionnaire asks about AI chatbots and web access; expect a **Teen**
rating for unrestricted internet/AI interaction. Blackjack is play-money
only (no real-money gambling, no purchases) — declare simulated gambling if
asked; it can add a "Simulated Gambling" descriptor.

## Declarations to be ready for

- **Foreground service (dataSync):** the gateway runs as a foreground service
  so it stays available; the Console asks for a short justification and may
  ask for a demo video showing the persistent notification and the feature it
  powers (the always-on local AI gateway).
- **Bluetooth:** `BLUETOOTH_CONNECT` is used to talk to the user's own OBD-II
  adapter; declare that purpose if the permissions form asks.
- **No** `REQUEST_INSTALL_PACKAGES`, **no** battery-exemption request in the
  Play build (removed by the flavor overlay) — nothing to declare there.

## Installing the direct APK without virus warnings

The "harmful app" / extra-warning screens on install come from Google Play
Protect on the phone, not from the app — every APK installed outside the Play
Store gets them, and an app cannot switch them off. Two legitimate fixes:

1. **Android Developer Verification** (for the direct APK): in the Android
   Developer Console, verify your identity (government ID; one-time $25 for a
   full account — the same fee as Play — or use the free limited-distribution
   account, up to 20 devices, no ID or fee) and register the package
   `com.mhexeract.app` with the release signing certificate below. Registered
   package + verified developer = the normal install experience, no scare
   screens for this certificate — and it future-proofs 2027, when unverified
   sideloads get a 24-hour "advanced flow" wait worldwide.
   - Release certificate SHA-256 (public — it ships inside every APK):
     `53b57eb8251c6528061e05766f424c41ee1e94801a76a9f944f7c1a19dd9cccc`
   - Never rotate or replace the release key: registration, updates, and the
     gateway's own signature check all pin this certificate.
2. **The Play listing itself**: apps installed from Google Play never show
   sideload warnings, and Play apps were registered automatically.

Until then, "More details → Install anyway" on the warning screen is the
normal path for your own phone.
