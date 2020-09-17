#!/usr/bin/env bash

function print_usage()
{
    cat << EOF

Usage: $(basename "$0") <PATH>

This script is used to generate PDF reports from markdown reports.
The generated PDF reports will be put in the same directory as the markdown reports.

Required arguments:
    <PATH> the path (file or directory) that contains the markdown reports.

How to install dependencies:
    sudo apt install pandoc
    sudo apt install texlive-latex-recommended

EOF
    exit 1
}

if [[ $# -lt 1 ]]; then
    echo "ERROR: <PATH> must be given"
    print_usage
fi

path=$1

if [[ -d ${path} ]]; then
    cd ${path}

    for markdown_file in ./*.md; do
        name=$(basename "${markdown_file}" .md)
        pandoc -f markdown-implicit_figures -o "${name}.pdf" "${name}.md"
    done

elif [[ -f ${path} ]]; then
    cd $(dirname ${path})
    name=$(basename "${path}" .md)
    pandoc -f markdown-implicit_figures -o "${name}.pdf" "${name}.md"

else
    echo "ERROR: '${path}' is not a valid path"
    print_usage
fi
