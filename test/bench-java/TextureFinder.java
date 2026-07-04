import texture.TextureProvider;

import java.util.ArrayList;
import java.util.List;

// Reference TextureFinder with only the formation swapped to the benchmark
// fixture (24 top-face blocks). Search loops are byte-for-byte the original.
public class TextureFinder extends Thread {

    public static final ArrayList<RotationInfo> formation = new ArrayList<>();
    private static final List<RotationInfo> topsAndBottoms = new ArrayList<>();
    private static final List<RotationInfo> sides = new ArrayList<>();

    static {
        formation.add(new RotationInfo(1, 0, 0, 0, false));
        formation.add(new RotationInfo(2, 0, 0, 2, false));
        formation.add(new RotationInfo(0, 0, 1, 2, false));
        formation.add(new RotationInfo(1, 0, 1, 3, false));
        formation.add(new RotationInfo(2, 0, 1, 3, false));
        formation.add(new RotationInfo(0, 0, 2, 3, false));
        formation.add(new RotationInfo(1, 0, 2, 3, false));
        formation.add(new RotationInfo(2, 0, 2, 0, false));
        formation.add(new RotationInfo(3, 2, 2, 1, false));
        formation.add(new RotationInfo(4, 2, 2, 2, false));
        formation.add(new RotationInfo(0, 0, 3, 1, false));
        formation.add(new RotationInfo(1, 0, 3, 2, false));
        formation.add(new RotationInfo(2, 0, 3, 1, false));
        formation.add(new RotationInfo(3, 2, 3, 2, false));
        formation.add(new RotationInfo(4, 2, 3, 1, false));
        formation.add(new RotationInfo(0, 0, 4, 1, false));
        formation.add(new RotationInfo(1, 0, 4, 1, false));
        formation.add(new RotationInfo(2, 0, 4, 0, false));
        formation.add(new RotationInfo(3, 2, 4, 2, false));
        formation.add(new RotationInfo(4, 2, 4, 0, false));
        formation.add(new RotationInfo(0, 0, 5, 3, false));
        formation.add(new RotationInfo(1, 0, 5, 3, false));
        formation.add(new RotationInfo(2, 0, 5, 0, false));
        formation.add(new RotationInfo(3, 2, 5, 3, false));

        for (RotationInfo info : formation) {
            if(info.isSide) {
                sides.add(info);
            } else {
                topsAndBottoms.add(info);
            }
        }
    }

    private final int startX;
    private final int endX;
    private final TextureProvider textureProvider;

    TextureFinder(int startX, int endX, TextureProvider textureProvider) {
        this.startX = startX;
        this.endX = endX;
        this.textureProvider = textureProvider;
    }

    public void run() {
        long first=System.currentTimeMillis();

        for(int x = startX; x <= endX; x++) {
            for (int z = Main.zMin; z <= Main.zMax; z++) {
                nextAttempt:
                for (int y = Main.yMin; y <= Main.yMax; y++) {
                    for (RotationInfo b : topsAndBottoms) {
                        if(b.rotation != textureProvider.getTexture(x + b.x, y+b.y, z+b.z, 4)) {
                            continue nextAttempt;
                        }
                    }
                    for (RotationInfo b : sides) {
                        if(b.rotation != textureProvider.getTexture(x + b.x, y+b.y, z+b.z, 2)) {
                            continue nextAttempt;
                        }
                    }

                    System.out.println("X: " + x + " Y: " + y + " Z: " + z);
                }
            }
        }
        System.out.println(((System.currentTimeMillis()-first)/1000) + " seconds");
    }
}
