package com.mhexeract.app;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;

/** In-app web browser: address bar, back/forward/reload, and a hand-off to the system browser. */
public class BrowserActivity extends Activity {
    public static final String EXTRA_URL = "url";
    private WebView web;
    private EditText address;
    private ProgressBar progress;
    private Button back, forward;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        float dp = getResources().getDisplayMetrics().density;
        int bg = Color.parseColor("#161d21"), fg = Color.parseColor("#e6ecea");

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.parseColor("#0f1417"));

        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setBackgroundColor(bg);
        int p = (int) (6 * dp);
        bar.setPadding(p, p, p, p);

        Button close = tool("✕", "Close browser", fg);
        back = tool("‹", "Back", fg);
        forward = tool("›", "Forward", fg);
        Button reload = tool("↻", "Reload", fg);
        Button external = tool("↗", "Open in system browser", fg);

        address = new EditText(this);
        address.setSingleLine(true);
        address.setTextColor(fg);
        address.setHintTextColor(Color.parseColor("#8a9a97"));
        address.setHint("Search or enter address");
        address.setTextSize(15);
        address.setImeOptions(EditorInfo.IME_ACTION_GO);
        address.setSelectAllOnFocus(true);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
        bar.addView(close);
        bar.addView(back);
        bar.addView(forward);
        bar.addView(address, lp);
        bar.addView(reload);
        bar.addView(external);

        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progress.setMax(100);
        progress.setVisibility(View.GONE);

        web = new WebView(this);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(false);
        s.setBuiltInZoomControls(true);
        s.setDisplayZoomControls(false);
        s.setLoadWithOverviewMode(true);
        s.setUseWideViewPort(true);
        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest req) {
                String scheme = req.getUrl().getScheme();
                if ("http".equals(scheme) || "https".equals(scheme)) return false;
                try { // mailto:, tel:, intent: and app links go to the system
                    startActivity(new Intent(Intent.ACTION_VIEW, req.getUrl()));
                } catch (Exception ignored) {
                }
                return true;
            }

            @Override
            public void onPageStarted(WebView view, String url, Bitmap favicon) {
                if (!address.hasFocus()) address.setText(url);
                updateNav();
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                if (!address.hasFocus()) address.setText(url);
                updateNav();
            }
        });
        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onProgressChanged(WebView view, int p) {
                progress.setProgress(p);
                progress.setVisibility(p < 100 ? View.VISIBLE : View.GONE);
            }
        });

        root.addView(bar, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        root.addView(progress, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, (int) (3 * dp)));
        root.addView(web, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));
        setContentView(root);

        close.setOnClickListener(v -> finish());
        back.setOnClickListener(v -> { if (web.canGoBack()) web.goBack(); });
        forward.setOnClickListener(v -> { if (web.canGoForward()) web.goForward(); });
        reload.setOnClickListener(v -> web.reload());
        external.setOnClickListener(v -> {
            String u = web.getUrl();
            if (u != null) {
                try {
                    startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(u)));
                } catch (Exception ignored) {
                }
            }
        });
        address.setOnEditorActionListener((v, actionId, ev) -> {
            boolean go = actionId == EditorInfo.IME_ACTION_GO
                    || (ev != null && ev.getKeyCode() == KeyEvent.KEYCODE_ENTER && ev.getAction() == KeyEvent.ACTION_DOWN);
            if (!go) return false;
            web.loadUrl(normalize(address.getText().toString()));
            address.clearFocus();
            InputMethodManager imm = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
            if (imm != null) imm.hideSoftInputFromWindow(address.getWindowToken(), 0);
            web.requestFocus();
            return true;
        });

        if (savedInstanceState != null) {
            web.restoreState(savedInstanceState);
        } else {
            String start = getIntent().getStringExtra(EXTRA_URL);
            if (start == null && getIntent().getData() != null) start = getIntent().getData().toString();
            web.loadUrl(normalize(start == null ? "https://duckduckgo.com/" : start));
        }
        updateNav();
    }

    private Button tool(String label, String description, int color) {
        Button b = new Button(this, null, android.R.attr.borderlessButtonStyle);
        b.setText(label);
        b.setTextSize(20);
        b.setTextColor(color);
        b.setContentDescription(description);
        b.setMinWidth(0);
        b.setMinimumWidth(0);
        int w = (int) (42 * getResources().getDisplayMetrics().density);
        b.setLayoutParams(new LinearLayout.LayoutParams(w, w));
        b.setPadding(0, 0, 0, 0);
        return b;
    }

    /** Address as typed, an https upgrade for plain http (cleartext is disabled), or a web search. */
    static String normalize(String q) {
        q = q == null ? "" : q.trim();
        if (q.isEmpty()) return "https://duckduckgo.com/";
        if (q.startsWith("https://")) return q;
        if (q.startsWith("http://")) return "https://" + q.substring(7);
        if (!q.contains(" ") && q.matches("^[\\w.-]+\\.[a-zA-Z]{2,}(:\\d+)?(/.*)?$")) return "https://" + q;
        return "https://duckduckgo.com/?q=" + Uri.encode(q);
    }

    private void updateNav() {
        if (back == null) return;
        back.setAlpha(web.canGoBack() ? 1f : 0.35f);
        forward.setAlpha(web.canGoForward() ? 1f : 0.35f);
    }

    @Override
    protected void onSaveInstanceState(Bundle out) {
        super.onSaveInstanceState(out);
        web.saveState(out);
    }

    @Override
    public void onBackPressed() {
        if (web.canGoBack()) web.goBack();
        else super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        web.destroy();
        super.onDestroy();
    }
}
