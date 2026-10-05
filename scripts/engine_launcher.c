/* Native UCI launcher for GUIs such as Banksia. Build to build/chess-zero-engine. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
int main(int argc, char **argv) {
    char executable[PATH_MAX], resolved[PATH_MAX], script[PATH_MAX];
#ifdef __APPLE__
    unsigned int size = sizeof(executable);
    if (_NSGetExecutablePath(executable, &size) != 0) return 1;
#else
    ssize_t n = readlink("/proc/self/exe", executable, sizeof(executable)-1);
    if (n < 0) return 1;
    executable[n] = '\0';
#endif
    if (!realpath(executable, resolved)) return 1;
    char *last = strrchr(resolved, '/'); if (!last) return 1; *last = '\0';
    last = strrchr(resolved, '/'); if (!last) return 1; *last = '\0';
    if (snprintf(script, sizeof(script), "%s/scripts/run_engine.sh", resolved) >= sizeof(script)) return 1;
    char **args = calloc((size_t)argc+1, sizeof(char*)); if (!args) return 1;
    args[0] = script;
    for (int i=1; i<argc; ++i) args[i] = argv[i];
    execv(script, args);
    perror("chess-zero launcher"); return 127;
}
