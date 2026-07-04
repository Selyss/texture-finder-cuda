import texture.TextureProvider;
import texture.VanillaTextures;

// Benchmark configuration of the reference tool: identical workload to the
// CUDA benchmarks (formation_a volume, modern textures). Thread count via
// -Dbench.threads=N so the shipped default (10) and a fully tuned run use
// the same build. Search logic is untouched.
public class Main {

    public static final int xMin = -175000, xMax = -75000;
    public static final int zMin = -75000, zMax = -25000;
    public static final int yMin = -64, yMax = -40;
    public static final int threads = Integer.getInteger("bench.threads", 10);
    public static final TextureProvider mode = new VanillaTextures();

    public static void main(String[] args) {

        int xtotal = xMax - xMin;
        int perX = xtotal/threads;

        for (int start = xMin; start < xMax; start+= perX+1) {
            TextureFinder a = new TextureFinder(start,start+perX, mode);
            a.start();
        }

    }


}
