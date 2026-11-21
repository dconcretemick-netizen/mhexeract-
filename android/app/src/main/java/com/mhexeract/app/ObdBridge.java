package com.mhexeract.app;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.bluetooth.BluetoothSocket;
import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.UUID;

/**
 * Bluetooth SPP link to an ELM327 adapter, called from the gateway's Python
 * through Chaquopy. All methods are static and blocking; the Python side
 * calls them from its own threads, never the UI thread.
 */
public class ObdBridge {
    private static final UUID SPP = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB");
    private static Context ctx;
    private static BluetoothSocket socket;
    private static InputStream in;
    private static OutputStream out;

    public static void init(Context c) {
        ctx = c.getApplicationContext();
    }

    private static BluetoothAdapter adapter() {
        if (ctx == null) return null;
        BluetoothManager m = (BluetoothManager) ctx.getSystemService(Context.BLUETOOTH_SERVICE);
        return m != null ? m.getAdapter() : null;
    }

    public static boolean available() {
        try {
            BluetoothAdapter a = adapter();
            return a != null && a.isEnabled();
        } catch (Exception e) {
            return false;
        }
    }

    /** Bonded (paired) devices as a JSON array of {name, address}. */
    public static String devices() {
        JSONArray arr = new JSONArray();
        try {
            BluetoothAdapter a = adapter();
            if (a != null) {
                for (BluetoothDevice d : a.getBondedDevices()) {
                    JSONObject o = new JSONObject();
                    o.put("name", d.getName() == null ? "" : d.getName());
                    o.put("address", d.getAddress());
                    arr.put(o);
                }
            }
        } catch (Exception ignored) {
        }
        return arr.toString();
    }

    public static boolean connect(String address) {
        close();
        try {
            BluetoothAdapter a = adapter();
            if (a == null) return false;
            BluetoothDevice d = a.getRemoteDevice(address);
            socket = d.createRfcommSocketToServiceRecord(SPP);
            socket.connect();
            in = socket.getInputStream();
            out = socket.getOutputStream();
            return true;
        } catch (Exception e) {
            close();
            return false;
        }
    }

    /** Send one command and return the reply text up to the ELM327 prompt. */
    public static String exchange(String cmd, int timeoutMs) {
        if (socket == null || in == null || out == null) return "";
        try {
            out.write((cmd + "\r").getBytes("US-ASCII"));
            out.flush();
            StringBuilder sb = new StringBuilder();
            long end = System.currentTimeMillis() + timeoutMs;
            while (System.currentTimeMillis() < end) {
                int avail = in.available();
                if (avail > 0) {
                    byte[] buf = new byte[avail];
                    int n = in.read(buf);
                    if (n <= 0) break;
                    sb.append(new String(buf, 0, n, "US-ASCII"));
                    if (sb.indexOf(">") >= 0) break;
                } else {
                    Thread.sleep(25);
                }
            }
            return sb.toString().replace(">", " ");
        } catch (Exception e) {
            return "";
        }
    }

    public static void close() {
        try {
            if (socket != null) socket.close();
        } catch (Exception ignored) {
        }
        socket = null;
        in = null;
        out = null;
    }
}
