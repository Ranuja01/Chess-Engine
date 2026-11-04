// LD_PRELOAD malloc census for _malloc_census.py: counts every malloc and keeps the call stack of every 1024th one; at exit
// writes the count and the kept stacks to $MALLOC_TRACE_OUT, each frame as an offset into the main executable ("-" for frames
// in other objects such as libstdc++'s operator new). backtrace() unwinds through .eh_frame, so it sees callers that a
// frame-pointer perf profile loses inside glibc.
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <execinfo.h>

extern void *__libc_malloc(size_t);
static __thread int in_hook;
static unsigned long calls;
#define DEPTH 6
#define MAXREC 400000
static void *rec[MAXREC][DEPTH];
static int nrec;

void *malloc(size_t n)
{
	void *p = __libc_malloc(n);
	if (!in_hook) {
		unsigned long c = __atomic_add_fetch(&calls, 1, __ATOMIC_RELAXED);
		if ((c & 1023) == 0) {
			in_hook = 1;   // backtrace() allocates on its first call
			void *bt[DEPTH + 1];
			int k = backtrace(bt, DEPTH + 1);
			int i = __atomic_fetch_add(&nrec, 1, __ATOMIC_RELAXED);
			if (i < MAXREC)
				for (int j = 0; j < DEPTH; ++j)
					rec[i][j] = (j + 1 < k) ? bt[j + 1] : 0;
			in_hook = 0;
		}
	}
	return p;
}

__attribute__((destructor)) static void dump(void)
{
	in_hook = 1;
	const char *out = getenv("MALLOC_TRACE_OUT");
	if (!out)
		return;
	// The executable's own mappings, to turn return addresses into file offsets addr2line understands.
	unsigned long lo[16], hi[16], off[16];
	int nm = 0;
	char exe[512] = {0};
	FILE *e = fopen("/proc/self/cmdline", "r");
	if (e) {
		size_t r = fread(exe, 1, sizeof exe - 1, e);
		exe[r] = 0;
		fclose(e);
	}
	const char *base = strrchr(exe, '/');
	base = base ? base + 1 : exe;
	FILE *m = fopen("/proc/self/maps", "r");
	char line[1024];
	while (m && fgets(line, sizeof line, m) && nm < 16) {
		unsigned long a, b, o;
		char perm[8];
		if (sscanf(line, "%lx-%lx %7s %lx", &a, &b, perm, &o) == 4 && strstr(line, base) && perm[2] == 'x') {
			lo[nm] = a;
			hi[nm] = b;
			off[nm] = o;
			++nm;
		}
	}
	if (m)
		fclose(m);
	FILE *f = fopen(out, "w");
	if (!f)
		return;
	int n = nrec < MAXREC ? nrec : MAXREC;
	fprintf(f, "calls %lu sampled %d exe %s\n", calls, n, base);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < DEPTH; ++j) {
			unsigned long a = (unsigned long)rec[i][j];
			int hit = -1;
			for (int t = 0; t < nm; ++t)
				if (a >= lo[t] && a < hi[t])
					hit = t;
			if (hit >= 0)
				fprintf(f, " %lx", a - lo[hit] + off[hit]);
			else
				fprintf(f, " -");
		}
		fputc('\n', f);
	}
	fclose(f);
}
