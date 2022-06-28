#!/bin/bash

readonly TUFFY_DIR='/tuffy'

readonly JAR_PATH="${TUFFY_DIR}/tuffy.jar"
readonly CONFIG_PATH="${TUFFY_DIR}/tuffy.conf"

readonly IO_DIR="${TUFFY_DIR}/io"
readonly PROGRAM_PATH="${IO_DIR}/prog.mln"
readonly EVIDENCE_PATH="${IO_DIR}/evidence.db"
readonly QUERY_PATH="${IO_DIR}/query.db"
readonly OUTPUT_PATH="${IO_DIR}/out.txt"

function check_files() {
    if [[ ! -f "${PROGRAM_PATH}" ]]; then
        echo "ERROR: Tuffy program (${PROGRAM_PATH}) does not exist."
        exit 101
    fi

    if [[ ! -f "${EVIDENCE_PATH}" ]]; then
        echo "ERROR: Tuffy evidence (${EVIDENCE_PATH}) does not exist."
        exit 102
    fi

    if [[ ! -f "${QUERY_PATH}" ]]; then
        echo "ERROR: Tuffy query (${QUERY_PATH}) does not exist."
        exit 103
    fi
}

function setup_postgres() {
    su postgres -c '/usr/local/bin/docker-entrypoint.sh postgres' &
    # TODO(eriq): Wait for init to complete.
    sleep 2
}

function run_tuffy() {
    java -jar "${JAR_PATH}" -conf "${CONFIG_PATH}" \
        -mln "${PROGRAM_PATH}" \
        -evidence "${EVIDENCE_PATH}" \
        -queryFile "${QUERY_PATH}" \
        -result "${OUTPUT_PATH}"

    echo "--- RESULTS ---"
    cat "${OUTPUT_PATH}"
}

function main() {
    if [[ $# -ne 0 ]]; then
        echo "USAGE: $0"
        exit 1
    fi

    set -e
    trap exit SIGINT

    check_files
    setup_postgres
    run_tuffy $@
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
