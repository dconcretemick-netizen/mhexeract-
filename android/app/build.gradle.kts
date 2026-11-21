plugins {
    id("com.android.application")
    id("com.chaquo.python")
}

// Bundle the shared gateway sources from the repository root into the app's Python
// directory, so the APK always runs the same gateway.py / models.json / dashboard.
val repoRoot = rootDir.parentFile
val pyDir = file("src/main/python")
copy {
    from(repoRoot) { include("gateway.py", "models.json", "dashboard.html", "game.html", "solitaire.html", "tetris.html", "platformer.html", "oregon.html", "blackjack.html", "news.json", "build_info.json", "dtc_codes.json", "dtc_codes.SOURCE.txt") }
    into(pyDir)
}

android {
    namespace = "com.mhexeract.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.mhexeract.app"
        minSdk = 26
        targetSdk = 35
        versionCode = (System.getenv("MCUBED_VERSION_CODE") ?: "1").toInt()
        versionName = System.getenv("MCUBED_VERSION_NAME") ?: "0.1.0"
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }

    buildFeatures {
        buildConfig = true
    }

    // Two distribution channels (project codename 6D):
    //  - direct: the GitHub/sideload build; it updates itself from GitHub releases.
    //  - play:   the Google Play build; Play handles updates, and the manifest overlay
    //            in src/play removes the permissions Play forbids for that (installing
    //            packages, battery-optimization exemption). The CHANNEL build config
    //            field tells the gateway which world it lives in.
    flavorDimensions += "channel"

    productFlavors {
        create("direct") {
            dimension = "channel"
            buildConfigField("String", "CHANNEL", "\"direct\"")
        }
        create("play") {
            dimension = "channel"
            buildConfigField("String", "CHANNEL", "\"play\"")
        }
    }

    // One permanent release key (from CI secrets) so each new APK installs as an update over the last one.
    val ks = System.getenv("MCUBED_KEYSTORE_FILE")
    signingConfigs {
        if (ks != null && file(ks).exists()) {
            create("mhexeract") {
                storeFile = file(ks)
                storeType = "pkcs12"
                storePassword = System.getenv("MCUBED_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("MCUBED_KEY_ALIAS") ?: "mcubed"
                keyPassword = System.getenv("MCUBED_KEYSTORE_PASSWORD")
                // Every signature scheme: v1 (JAR) for older tools and scanners, v2 for Android 7+,
                // v3 so the key can be rotated later without breaking updates.
                enableV1Signing = true
                enableV2Signing = true
                enableV3Signing = true
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Local builds without the key fall back to the debug key.
            signingConfig = signingConfigs.findByName("mhexeract") ?: signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

chaquopy {
    defaultConfig {
        version = "3.13"
    }
}
