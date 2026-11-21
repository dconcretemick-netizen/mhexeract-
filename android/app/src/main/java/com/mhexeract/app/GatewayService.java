package com.mhexeract.app;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.IBinder;

/**
 * Keep-alive foreground service: while it runs, Android is far less likely to
 * kill the app process, so the embedded gateway stays reachable in the
 * background. The gateway itself still runs in MainActivity's process; this
 * service only holds the process open and shows the required notification.
 */
public class GatewayService extends Service {
    private static final String CHANNEL = "mhexeract_gateway";
    private static final int NOTIFICATION_ID = 42;

    public static void start(Context ctx) {
        Intent i = new Intent(ctx, GatewayService.class);
        if (Build.VERSION.SDK_INT >= 26) ctx.startForegroundService(i);
        else ctx.startService(i);
    }

    public static void stop(Context ctx) {
        ctx.stopService(new Intent(ctx, GatewayService.class));
    }

    @Override
    public void onCreate() {
        super.onCreate();
        if (Build.VERSION.SDK_INT >= 26) {
            NotificationManager nm = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
            if (nm != null && nm.getNotificationChannel(CHANNEL) == null) {
                nm.createNotificationChannel(new NotificationChannel(
                        CHANNEL, "MHexeract gateway", NotificationManager.IMPORTANCE_LOW));
            }
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Intent open = new Intent(this, MainActivity.class)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        int piFlags = PendingIntent.FLAG_UPDATE_CURRENT
                | (Build.VERSION.SDK_INT >= 31 ? PendingIntent.FLAG_IMMUTABLE : 0);
        int icon = getApplicationInfo().icon != 0 ? getApplicationInfo().icon
                : android.R.drawable.stat_notify_sync;
        Notification.Builder b = Build.VERSION.SDK_INT >= 26
                ? new Notification.Builder(this, CHANNEL)
                : new Notification.Builder(this);
        Notification n = b.setContentTitle("MHexeract is running")
                .setContentText("Your AI gateway is available in the background. Tap to open.")
                .setSmallIcon(icon)
                .setContentIntent(PendingIntent.getActivity(this, 0, open, piFlags))
                .setOngoing(true)
                .build();
        startForeground(NOTIFICATION_ID, n);
        return START_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
