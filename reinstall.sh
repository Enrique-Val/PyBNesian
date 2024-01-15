#!/usr/bin/env
# workon aiTanium-master # This doesn't work
export CC="ccache gcc"
python setup.py clean --all
time python setup.py install
# python setup.py develop # For verbose output
# export CC="ccache clang-14"

# export LDSHARED="clang-14 -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fwrapv -O2" # NOTE: Ignored?
# source venv/bin/activate