# The valgrind side of the boilerplate: macOS builds no valgrind on Apple
# Silicon, so every callgrind/memcheck run happens inside this Linux image
# (OrbStack/Docker), with the project mounted at /repo. rustc and cargo are
# included so a run can build the target for Linux first (`build:` argument
# of valgrind_run); container builds go to .boiler/target-linux so they never
# clobber the host's target/.
FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends rustc cargo valgrind \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /repo