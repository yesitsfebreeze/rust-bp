# The valgrind side of the boilerplate: macOS builds no valgrind on Apple
# Silicon, so every callgrind/memcheck run happens inside this Linux image
# (OrbStack/Docker), with the project mounted at /repo. The rustup stable
# toolchain lets a run build the target for Linux first (`build:` argument of
# valgrind_run); container builds go to <daemon dir>/target-linux (the host's
# daemon dir is mounted at its own path) so they never clobber the host's
# target/.
FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates valgrind \
    && rm -rf /var/lib/apt/lists/*
RUN set -eux; \
    export RUSTUP_HOME=/usr/local/rustup; \
    export CARGO_HOME=/usr/local/cargo; \
    curl https://sh.rustup.rs -sSf -o /tmp/rustup.sh; \
    sh /tmp/rustup.sh -y --profile minimal --default-toolchain stable; \
    rm -rf /usr/local/rustup/tmp /tmp/rustup.sh; \
    /usr/local/cargo/bin/rustc --version; valgrind --version
ENV PATH=/usr/local/cargo/bin:$PATH RUSTUP_HOME=/usr/local/rustup
WORKDIR /repo