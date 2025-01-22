#!/bin/bash

# Find all Python files and add the encoding declaration
for file in $(find . -type f -name "*.py"); do
    if ! grep -q '# -*- coding: utf-8 -*-' "$file"; then
        echo '# -*- coding: utf-8 -*-' | cat - "$file" > temp && mv temp "$file"
    fi
done
