# Register the MHexeract APK with Android

Android developer verification links an app's package name and signing key to a verified developer. In 2026 it only applies to apps installed from participating stores in Brazil, Indonesia, Singapore and Thailand. Sideloaded APKs are covered when the global rollout starts in 2027. Registering early keeps "Download and install" working without extra warnings.

Official guide: https://developer.android.com/developer-verification/guides/android-developer-console

## Values to enter

| Field | Value |
|---|---|
| Package name | `com.mhexeract.app` |
| Friendly name | `MHexeract` |
| SHA-256 certificate fingerprint | `53:B5:7E:B8:25:1C:65:28:06:1E:05:76:6F:42:4C:41:EE:1E:94:80:1A:76:A9:F9:44:F7:C1:A1:9D:D9:CC:CC` |
| Certificate subject | `CN=MHexeract, O=dconcretemick-netizen` |

## Free of cost

Choose **Personal** and **Limited distribution**. It has no fee and needs no government ID. It needs a Google account with 2-Step Verification, a payments profile and a contact email, and allows up to 20 devices, which join by accepting an invitation.

Do not choose Full distribution. It has a one-time 25 USD fee, which breaks the MHexeract free-of-cost rule.

## Steps

1. Sign in to the Android Developer Console with your Google account: https://android.google.com/developerconsole.
2. Create a Personal, Limited distribution account and finish verification.
3. Open **Package names**, then **Register package name**. Enter the package name and friendly name from the table.
4. Select **Add key**.
   - If the console lists eligible fingerprints, pick the one in the table. MHexeract has been installed before, so this is the likely case.
   - Otherwise paste the fingerprint from the table.
5. If the console asks for proof of ownership, copy the snippet it shows. Then:
   1. In GitHub, open the MHexeract repo, then **Settings**, **Secrets and variables**, **Actions**, **Variables**.
   2. Add a repository variable named `ADI_REGISTRATION` and paste the snippet as its value. The snippet is not a password.
   3. Open **Actions**, then **Build Android APK**, then **Run workflow** on `main`. CI puts the snippet in `assets/adi-registration.properties` and signs the APK with the MHexeract key.
   4. Download `MCUBED.apk` from the install page, https://github.com/dconcretemick-netizen/MCUBED/releases/tag/latest, and upload it in the console.
6. Wait for the email. The package status changes to **Registered**.

You can leave `ADI_REGISTRATION` in place. It's harmless, and it lets you redo the proof later.

## Keep the key safe

Registration is tied to the MHexeract signing key. If the key is lost, the package can't be registered or updated under the same name, so keep the backup of the keystore.
