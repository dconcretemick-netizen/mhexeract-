package com.mcubed.app;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.PendingIntent;
import android.content.Intent;
import android.content.pm.PackageInstaller;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.PowerManager;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.text.InputType;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuItem;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;

import org.json.JSONObject;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

/** Hosts the MHexeract gateway (embedded Python) and shows its dashboard full screen. */
public class MainActivity extends Activity {
    private static final String PREFS = "mcubed";
    private static final String KEY_CONFIGURED = "keys_configured";
    private static final String KEY_KEEP_BG = "keep_background";
    /** Provider key slots: environment variable name and label shown in the dialog. */
    private static final String[][] KEYS = {
            {"GEMINI_API_KEY", "Google Gemini (free)"},
            {"GROQ_API_KEY", "Groq (free)"},
            {"OPENROUTER_API_KEY", "OpenRouter (free models only unless paid AI is on)"},
            {"MISTRAL_API_KEY", "Mistral (free)"},
            {"SAMBANOVA_API_KEY", "SambaNova (free)"},
            {"NVIDIA_API_KEY", "NVIDIA NIM (free credits)"},
            {"HF_TOKEN", "Hugging Face (free credits)"},
            {"COHERE_API_KEY", "Cohere (free trial key)"},
            {"GITHUB_TOKEN", "GitHub token (optional: repositories)"},
            {"OPENAI_API_KEY", "OpenAI (paid, off unless paid AI is on)"},
            {"DEEPSEEK_API_KEY", "DeepSeek (paid, off unless paid AI is on)"},
            {"CEREBRAS_API_KEY", "Cerebras (paid)"},
            {"XAI_API_KEY", "xAI Grok (paid)"},
            {"TOGETHER_API_KEY", "Together AI (paid)"},
            {"FIREWORKS_API_KEY", "Fireworks AI (paid)"},
            {"DEEPINFRA_API_KEY", "DeepInfra (paid)"},
            {"NOVITA_API_KEY", "Novita AI (paid)"},
            {"HYPERBOLIC_API_KEY", "Hyperbolic (paid)"},
            {"MOONSHOT_API_KEY", "Moonshot Kimi (paid)"},
            {"DASHSCOPE_API_KEY", "Alibaba Qwen (paid)"},
            {"ZAI_API_KEY", "Z.ai GLM (paid)"},
            {"MINIMAX_API_KEY", "MiniMax (paid)"},
            {"MODEL_API_KEY", "Meta Muse AI (paid)"},
    };
    private static final int MENU_KEY = 1, MENU_RELOAD = 2, MENU_TOOLBOX = 3, MENU_CLOSE = 4;

    private WebView web;
    private TextView status;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ObdBridge.init(this);
        if (Build.VERSION.SDK_INT >= 31
                && checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.BLUETOOTH_CONNECT}, 7);
        }
        status = new TextView(this);
        status.setText("Starting MHexeract gateway…");
        status.setTextColor(Color.parseColor("#9aa8a6"));
        status.setGravity(Gravity.CENTER);
        setContentView(status);

        web = new WebView(this);
        web.setBackgroundColor(Color.parseColor("#0f1417"));
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(false);
        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest req) {
                Uri u = req.getUrl();
                if ("127.0.0.1".equals(u.getHost())) return false;
                String scheme = u.getScheme();
                if ("mhexeract".equals(scheme)) { // mhexeract://install-update from the Update tab
                    if ("install-update".equals(u.getHost())) installUpdate();
                    if ("close-app".equals(u.getHost())) confirmClose(); // dashboard's Close app button
                    return true;
                }
                if ("github.com".equals(u.getHost()) && u.getPath() != null && u.getPath().contains("/releases")) {
                    startActivity(new Intent(Intent.ACTION_VIEW, u)); // release downloads need the system browser
                    return true;
                }
                if ("http".equals(scheme) || "https".equals(scheme)) { // web pages open in the in-app browser
                    startActivity(new Intent(MainActivity.this, BrowserActivity.class)
                            .putExtra(BrowserActivity.EXTRA_URL, u.toString()));
                } else {
                    try {
                        startActivity(new Intent(Intent.ACTION_VIEW, u));
                    } catch (Exception ignored) {
                    }
                }
                return true;
            }
        });

        if (savedInstanceState == null && !prefs().getBoolean(KEY_CONFIGURED, false)) {
            askForKey(true);
        } else {
            startGateway();
        }
        handleInstallStatus(getIntent());
    }

    private SharedPreferences prefs() {
        return getSharedPreferences(PREFS, MODE_PRIVATE);
    }

    private void startGateway() {
        applyBackgroundPref();
        final String keys = keysJson();
        new Thread(() -> {
            try {
                if (!Python.isStarted()) Python.start(new AndroidPlatform(this));
                final String url = Python.getInstance().getModule("android_main")
                        .callAttr("start", getFilesDir().getAbsolutePath() + "/gateway", keys, BuildConfig.CHANNEL).toString();
                runOnUiThread(() -> {
                    setContentView(web);
                    if (web.getUrl() == null) web.loadUrl(url);
                });
            } catch (Throwable t) {
                runOnUiThread(() -> status.setText("MHexeract failed to start:\n" + t.getMessage()));
            }
        }, "mhexeract-start").start();
    }

    private File keysFile() {
        return new File(getFilesDir(), "gateway/keys.json");
    }

    /** Saved keys as one object: the dashboard's keys.json first, then this dialog's prefs on top. */
    private JSONObject mergedKeys() {
        JSONObject o = new JSONObject();
        try {
            File f = keysFile();
            if (f.isFile()) {
                byte[] buf = new byte[(int) f.length()];
                try (FileInputStream in = new FileInputStream(f)) {
                    int n = 0, r;
                    while (n < buf.length && (r = in.read(buf, n, buf.length - n)) > 0) n += r;
                }
                JSONObject fromFile = new JSONObject(new String(buf, "UTF-8"));
                for (String[] k : KEYS) {
                    String v = fromFile.optString(k[0], "");
                    if (!v.isEmpty()) o.put(k[0], v);
                }
            }
        } catch (Exception ignored) {
        }
        try {
            for (String[] k : KEYS) {
                String v = prefs().getString(k[0], "");
                if (!v.isEmpty()) o.put(k[0], v);
            }
        } catch (Exception ignored) {
        }
        return o;
    }

    private String keysJson() {
        return mergedKeys().toString();
    }

    private boolean hasKey(String name) {
        return !mergedKeys().optString(name, "").isEmpty();
    }

    private int savedKeyCount() {
        JSONObject o = mergedKeys();
        int n = 0;
        for (String[] k : KEYS) if (!o.optString(k[0], "").isEmpty()) n++;
        return n;
    }

    private void askForKey(final boolean firstRun) {
        final LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        int pad = (int) (20 * getResources().getDisplayMetrics().density);
        box.setPadding(pad, pad / 2, pad, 0);
        final EditText[] fields = new EditText[KEYS.length];
        for (int i = 0; i < KEYS.length; i++) {
            TextView label = new TextView(this);
            label.setText(KEYS[i][1]);
            box.addView(label);
            EditText input = new EditText(this);
            input.setSingleLine(true);
            input.setHint(hasKey(KEYS[i][0]) ? "saved (leave blank to keep)" : "not set");
            input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
            box.addView(input);
            fields[i] = input;
        }
        // A Save button inside the form itself, at the end of the list: the dialog's
        // button bar is easy to miss behind the keyboard on a long form like this.
        final Button saveBtn = new Button(this);
        saveBtn.setText("Save keys");
        box.addView(saveBtn);
        final android.widget.ScrollView scroll = new android.widget.ScrollView(this);
        scroll.addView(box);
        final AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle("AI provider keys")
                .setMessage("MHexeract is free: only free AI is used unless you turn on paid AI. Keys are stored only on this device. "
                        + "With no keys, MHexeract runs in demo mode with simulated replies.")
                .setView(scroll)
                .setCancelable(false)
                .setPositiveButton("Save keys", (d, w) -> saveKeysFrom(fields, firstRun))
                .setNegativeButton(firstRun ? "Demo mode" : "Cancel", (d, w) -> {
                    if (firstRun) {
                        prefs().edit().putBoolean(KEY_CONFIGURED, true).apply();
                        applyKeys(true);
                    }
                })
                .setNeutralButton(firstRun ? null : "Remove all", firstRun ? null : (d, w) -> {
                    SharedPreferences.Editor e = prefs().edit();
                    for (String[] k : KEYS) e.remove(k[0]);
                    e.apply();
                    keysFile().delete(); // the dashboard saves to the same store
                    applyKeys(false);
                })
                .create();
        saveBtn.setOnClickListener(v -> {
            saveKeysFrom(fields, firstRun);
            dialog.dismiss();
        });
        dialog.show();
    }

    /** Persist whatever was typed (blank fields keep the saved key) and say so out loud. */
    private void saveKeysFrom(EditText[] fields, boolean firstRun) {
        SharedPreferences.Editor e = prefs().edit();
        for (int i = 0; i < KEYS.length; i++) {
            String v = fields[i].getText().toString().trim();
            if (!v.isEmpty()) e.putString(KEYS[i][0], v);
        }
        e.putBoolean(KEY_CONFIGURED, true).apply();
        int saved = savedKeyCount();
        Toast.makeText(this, saved > 0 ? "AI keys saved (" + saved + ")" : "No keys saved \u2014 demo mode",
                Toast.LENGTH_LONG).show();
        applyKeys(firstRun);
    }

    private void applyKeys(boolean firstRun) {
        if (firstRun) {
            startGateway();
            return;
        }
        final String keys = keysJson();
        new Thread(() -> {
            try {
                if (Python.isStarted()) Python.getInstance().getModule("android_main").callAttr("set_keys", keys);
                runOnUiThread(() -> web.reload());
            } catch (Throwable ignored) {
            }
        }).start();
    }

    /** Start or stop the keep-alive foreground service, as the Toolbox toggle says. */
    private void applyBackgroundPref() {
        boolean keep = prefs().getBoolean(KEY_KEEP_BG, true);
        if (keep) {
            if (Build.VERSION.SDK_INT >= 33
                    && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 7);
            }
            GatewayService.start(this);
        } else {
            GatewayService.stop(this);
        }
    }

    private boolean batteryUnrestricted() {
        if (Build.VERSION.SDK_INT < 23) return true;
        PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
        return pm != null && pm.isIgnoringBatteryOptimizations(getPackageName());
    }

    /** Basic background behaviour, in one place: keep-alive, battery, status, restart. */
    private void showToolbox() {
        final LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        int pad = (int) (20 * getResources().getDisplayMetrics().density);
        box.setPadding(pad, pad / 2, pad, pad / 2);

        TextView statusLine = new TextView(this);
        statusLine.setText("Gateway: running in this app\nAI keys saved: " + savedKeyCount()
                + "\nBattery: " + (batteryUnrestricted() ? "unrestricted (good)" : "restricted \u2014 Android may stop MHexeract"));
        box.addView(statusLine);

        final CheckBox keep = new CheckBox(this);
        keep.setText("Keep running in the background");
        keep.setChecked(prefs().getBoolean(KEY_KEEP_BG, true));
        keep.setOnCheckedChangeListener((b, on) -> {
            prefs().edit().putBoolean(KEY_KEEP_BG, on).apply();
            applyBackgroundPref();
            Toast.makeText(this, on ? "MHexeract will keep running in the background"
                    : "Background keep-alive off", Toast.LENGTH_LONG).show();
        });
        box.addView(keep);

        Button battery = new Button(this);
        battery.setText("Allow background battery use");
        battery.setOnClickListener(v -> {
            try {
                startActivity(new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                        Uri.parse("package:" + getPackageName())));
            } catch (Exception ex) {
                try {
                    startActivity(new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS));
                } catch (Exception ignored) {
                }
            }
        });
        box.addView(battery);

        Button restart = new Button(this);
        restart.setText("Restart app");
        restart.setOnClickListener(v -> recreate());
        box.addView(restart);

        Button reload = new Button(this);
        reload.setText("Reload dashboard");
        reload.setOnClickListener(v -> {
            if (web != null) web.reload();
        });
        box.addView(reload);

        Button close = new Button(this);
        close.setText("Shut down MHexeract");
        close.setOnClickListener(v -> confirmClose());
        box.addView(close);

        new AlertDialog.Builder(this)
                .setTitle("Toolbox")
                .setView(box)
                .setPositiveButton("Close", null)
                .show();
    }

    /** Ask first, then shut everything down: service, gateway and app. */
    private void confirmClose() {
        new AlertDialog.Builder(this)
                .setTitle("Shut down MHexeract?")
                .setMessage("The gateway will stop and the app will close. Your keys and settings stay saved — just open MHexeract again to restart it.")
                .setPositiveButton("Shut down", (d, w) -> closeApp())
                .setNegativeButton("Cancel", null)
                .show();
    }

    /** Full stop: the keep-alive service ends, the task is removed, and the
     *  process exits — which also ends the embedded Python gateway with it. */
    private void closeApp() {
        GatewayService.stop(this);
        finishAndRemoveTask();
        android.os.Process.killProcess(android.os.Process.myPid());
        System.exit(0);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        menu.add(0, MENU_KEY, 0, "AI keys");
        menu.add(0, MENU_TOOLBOX, 1, "Toolbox");
        menu.add(0, MENU_RELOAD, 2, "Reload");
        menu.add(0, MENU_CLOSE, 3, "Close app");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == MENU_KEY) {
            askForKey(false);
            return true;
        }
        if (item.getItemId() == MENU_TOOLBOX) {
            showToolbox();
            return true;
        }
        if (item.getItemId() == MENU_RELOAD) {
            web.reload();
            return true;
        }
        if (item.getItemId() == MENU_CLOSE) {
            confirmClose();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onBackPressed() {
        if (web != null && web.canGoBack()) web.goBack();
        else super.onBackPressed();
    }

    private static final String ACTION_INSTALL_STATUS = "com.mcubed.app.INSTALL_STATUS";

    /** Install the APK the gateway downloaded (files/gateway/update.apk) as an in-place update. */
    private void installUpdate() {
        final File apk = new File(getFilesDir(), "gateway/update.apk");
        if (!apk.isFile()) {
            Toast.makeText(this, "Download the update first", Toast.LENGTH_LONG).show();
            return;
        }
        if (!getPackageManager().canRequestPackageInstalls()) {
            Toast.makeText(this, "Allow MHexeract to install updates, then tap Install again", Toast.LENGTH_LONG).show();
            startActivity(new Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:" + getPackageName())));
            return;
        }
        new Thread(() -> {
            try {
                PackageInstaller pi = getPackageManager().getPackageInstaller();
                PackageInstaller.SessionParams params =
                        new PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL);
                params.setAppPackageName(getPackageName());
                int id = pi.createSession(params);
                PackageInstaller.Session session = pi.openSession(id);
                try (InputStream in = new FileInputStream(apk);
                     OutputStream out = session.openWrite("mhexeract.apk", 0, apk.length())) {
                    byte[] buf = new byte[1 << 16];
                    int n;
                    while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
                    session.fsync(out);
                }
                Intent cb = new Intent(this, MainActivity.class).setAction(ACTION_INSTALL_STATUS)
                        .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP);
                int flags = PendingIntent.FLAG_UPDATE_CURRENT
                        | (Build.VERSION.SDK_INT >= 31 ? PendingIntent.FLAG_MUTABLE : 0);
                session.commit(PendingIntent.getActivity(this, 7, cb, flags).getIntentSender());
                session.close();
            } catch (Exception e) {
                final String msg = e.getMessage();
                runOnUiThread(() -> Toast.makeText(this, "Update failed: " + msg, Toast.LENGTH_LONG).show());
            }
        }, "mhexeract-install").start();
    }

    private void handleInstallStatus(Intent intent) {
        if (intent == null || !ACTION_INSTALL_STATUS.equals(intent.getAction())) return;
        int status = intent.getIntExtra(PackageInstaller.EXTRA_STATUS, PackageInstaller.STATUS_FAILURE);
        if (status == PackageInstaller.STATUS_PENDING_USER_ACTION) {
            @SuppressWarnings("deprecation")
            Intent confirm = intent.getParcelableExtra(Intent.EXTRA_INTENT);
            if (confirm != null) startActivity(confirm.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
        } else if (status != PackageInstaller.STATUS_SUCCESS) {
            String m = intent.getStringExtra(PackageInstaller.EXTRA_STATUS_MESSAGE);
            Toast.makeText(this, "Update not installed" + (m != null ? ": " + m : ""), Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        handleInstallStatus(intent);
    }

    @Override
    protected void onDestroy() {
        if (web != null) web.destroy();
        super.onDestroy();
    }
}
