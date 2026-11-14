#include <cstdio>
#include <random>
#include "texture.cuh"
static uint64_t P(uint64_t u){ return u*u*42317861ull + u*11ull; }
int main(){
  std::mt19937_64 r(7); long n=20000000; long bad[6]={0};
  for(long i=0;i<n;i++){
    uint64_t u=r(), hi=r();
    // legacy: parity from u mod 2^33, "is it 0 / is it 2" from u mod 2^34
    uint64_t a=(u&((1ull<<33)-1))|(hi<<33), b=(u&((1ull<<34)-1))|(hi<<34);
    int v=legacyFromCoordRandom((int64_t)P(u)>>16,4);
    int va=legacyFromCoordRandom((int64_t)P(a)>>16,4), vb=legacyFromCoordRandom((int64_t)P(b)>>16,4);
    bad[0]+= (v&1)!=(va&1);
    bad[1]+= (v%2==0||vb%2==0) && v!=vb;      // even values fully determined by low 34 bits
    // vanilla12: whole value from u mod 2^32, parity from u mod 2^17, even values from u mod 2^18
    uint64_t c=(u&0xffffffffull)|(hi<<32), d=(u&((1ull<<17)-1))|(hi<<17), e=(u&((1ull<<18)-1))|(hi<<18);
    int w=vanilla12FromMix((int64_t)P(u),4);
    bad[2]+= w!=vanilla12FromMix((int64_t)P(c),4);
    bad[3]+= (w&1)!=(vanilla12FromMix((int64_t)P(d),4)&1);
    int we=vanilla12FromMix((int64_t)P(e),4); bad[4]+= (w%2==0||we%2==0)&&w!=we;
    // modern: does truncating to the low 62 input bits ever leave the value intact? (control: expect ~75% change)
    uint64_t g=(u&((1ull<<62)-1))|(hi<<62);
    bad[5]+= modernFromCoordRandom((int64_t)P(u)>>16,4)!=modernFromCoordRandom((int64_t)P(g)>>16,4);
  }
  printf("legacy parity vs low33: %ld bad\nlegacy even-values vs low34: %ld bad\nvanilla12 vs low32: %ld bad\nvanilla12 parity vs low17: %ld bad\nvanilla12 even-values vs low18: %ld bad\nmodern changes when only top 2 input bits change: %.1f%%\n",bad[0],bad[1],bad[2],bad[3],bad[4],100.0*bad[5]/n);
}
