import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

import texture.Sodium19Textures;
import texture.SodiumTextures;
import texture.TextureProvider;
import texture.Vanilla12Textures;
import texture.Vanilla21_1Textures;
import texture.VanillaTextures;

/**
 * Ground-truth dump driver. Compiled against the UNMODIFIED texture classes
 * from 19MisterX98/TextureRotations (copied in by regen.sh), so every dumped
 * value is produced by the reference implementation running on a JVM.
 *
 * Dumps are deterministic and platform-independent (verified byte-identical
 * across JDK 17/Linux/x86_64 and JDK 25/macOS/ARM). Formats consumed by
 * test/oracle_diff.cpp:
 *
 *   {legacy,modern}_{top,side}.bin  one raw byte per coordinate,
 *       offset = (yIndex*1024 + z+512)*1024 + x+512,
 *       yIndex over {-64,-54,0,63,255,319}, z,x in [-512,511]
 *   extremes.txt / random.txt  lines: "x y z legacy_top legacy_side modern_top modern_side"
 *
 * Usage:
 *   java Dump check              fixture gate (24/24 or exit 1)
 *   java Dump query <x> <y> <z>  print one line in the text format
 *   java Dump dumps <outdir>     run the gate, then generate all dumps
 */
public class Dump {

    static final TextureProvider LEGACY = new Vanilla21_1Textures();
    static final TextureProvider MODERN = new VanillaTextures();
    static final TextureProvider VANILLA12 = new Vanilla12Textures();
    static final TextureProvider SODIUM = new SodiumTextures();
    static final TextureProvider SODIUM19 = new Sodium19Textures();

    static final int[] GRID_Y = {-64, -54, 0, 63, 255, 319};
    static final int[] EXTREME_Y = {-64, 0, 319};

    // Real-world verified formation (test/fixtures/formation_a.txt): dx dy dz expected.
    static final int[][] FIXTURE = {
        {1, 0, 0, 0}, {2, 0, 0, 2}, {0, 0, 1, 2}, {1, 0, 1, 3}, {2, 0, 1, 3},
        {0, 0, 2, 3}, {1, 0, 2, 3}, {2, 0, 2, 0}, {3, 2, 2, 1}, {4, 2, 2, 2},
        {0, 0, 3, 1}, {1, 0, 3, 2}, {2, 0, 3, 1}, {3, 2, 3, 2}, {4, 2, 3, 1},
        {0, 0, 4, 1}, {1, 0, 4, 1}, {2, 0, 4, 0}, {3, 2, 4, 2}, {4, 2, 4, 0},
        {0, 0, 5, 3}, {1, 0, 5, 3}, {2, 0, 5, 0}, {3, 2, 5, 3}};
    static final int FX = -108723, FY = -54, FZ = -69736;

    public static void main(String[] args) throws IOException {
        if (args.length == 1 && args[0].equals("check")) {
            System.exit(check() ? 0 : 1);
        } else if (args.length == 4 && args[0].equals("query")) {
            int x = Integer.parseInt(args[1]);
            int y = Integer.parseInt(args[2]);
            int z = Integer.parseInt(args[3]);
            System.out.println(line(x, y, z));
        } else if (args.length == 2 && args[0].equals("dumps")) {
            if (!check()) {
                System.err.println("fixture gate FAILED; refusing to generate dumps");
                System.exit(1);
            }
            generate(new File(args[1]));
        } else {
            System.err.println("Usage: Dump check | query <x> <y> <z> | dumps <outdir>");
            System.exit(2);
        }
    }

    static boolean check() {
        int ok = 0;
        for (int[] b : FIXTURE)
            if (MODERN.getTexture(FX + b[0], FY + b[1], FZ + b[2], 4) == b[3])
                ok++;
        System.out.println("fixture gate: " + ok + "/" + FIXTURE.length);
        return ok == FIXTURE.length;
    }

    static String line(int x, int y, int z) {
        return x + " " + y + " " + z + " "
                + LEGACY.getTexture(x, y, z, 4) + " " + LEGACY.getTexture(x, y, z, 2) + " "
                + MODERN.getTexture(x, y, z, 4) + " " + MODERN.getTexture(x, y, z, 2) + " "
                + VANILLA12.getTexture(x, y, z, 4) + " " + VANILLA12.getTexture(x, y, z, 2) + " "
                + SODIUM.getTexture(x, y, z, 4) + " " + SODIUM.getTexture(x, y, z, 2) + " "
                + SODIUM19.getTexture(x, y, z, 4) + " " + SODIUM19.getTexture(x, y, z, 2);
    }

    static void generate(File dir) throws IOException {
        dir.mkdirs();

        grid(dir, "legacy_top.bin", LEGACY, 4);
        grid(dir, "legacy_side.bin", LEGACY, 2);
        grid(dir, "modern_top.bin", MODERN, 4);
        grid(dir, "modern_side.bin", MODERN, 2);
        grid(dir, "vanilla12_top.bin", VANILLA12, 4);
        grid(dir, "vanilla12_side.bin", VANILLA12, 2);
        grid(dir, "sodium_top.bin", SODIUM, 4);
        grid(dir, "sodium_side.bin", SODIUM, 2);
        grid(dir, "sodium19_top.bin", SODIUM19, 4);
        grid(dir, "sodium19_side.bin", SODIUM19, 2);

        // 32 values: {-30000000..-29999985} then {29999985..30000000}
        int[] ex = new int[32];
        for (int i = 0; i < 16; i++) ex[i] = -30000000 + i;
        for (int i = 0; i < 16; i++) ex[16 + i] = 29999985 + i;
        try (PrintWriter w = new PrintWriter(new File(dir, "extremes.txt"))) {
            for (int y : EXTREME_Y)
                for (int x : ex)
                    for (int z : ex)
                        w.println(line(x, y, z));
        }

        Random r = new Random(12345);
        try (PrintWriter w = new PrintWriter(new File(dir, "random.txt"))) {
            for (int i = 0; i < 1000000; i++) {
                int x = -30000000 + r.nextInt(60000001);
                int z = -30000000 + r.nextInt(60000001);
                int y = -64 + r.nextInt(384);
                w.println(line(x, y, z));
            }
        }

        try (PrintWriter w = new PrintWriter(new File(dir, "README.md"))) {
            w.println("# Oracle dumps");
            w.println();
            w.println("Generated by test/oracle/Dump.java from the reference implementation");
            w.println("(19MisterX98/TextureRotations texture classes, unmodified).");
            w.println();
            w.println("- `{legacy,modern}_{top,side}.bin`: one raw byte per coordinate;");
            w.println("  offset = (yIndex*1024 + z+512)*1024 + x+512;");
            w.println("  yIndex over {-64,-54,0,63,255,319}; z,x in [-512,511].");
            w.println("- `extremes.txt`, `random.txt`: lines \"x y z\" followed by top/side pairs");
            w.println("  for legacy, modern, vanilla12, sodium, sodium19 (10 value columns).");
            w.println("- random.txt: java.util.Random(12345); per coordinate, in order:");
            w.println("  x = -30000000 + nextInt(60000001); z likewise; y = -64 + nextInt(384).");
        }
        System.out.println("dumps written to " + dir);
    }

    static void grid(File dir, String name, TextureProvider p, int mod) throws IOException {
        try (BufferedOutputStream out = new BufferedOutputStream(
                new FileOutputStream(new File(dir, name)), 1 << 20)) {
            for (int y : GRID_Y)
                for (int z = -512; z <= 511; z++)
                    for (int x = -512; x <= 511; x++)
                        out.write(p.getTexture(x, y, z, mod));
        }
    }
}
