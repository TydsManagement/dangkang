#!/bin/bash
set -e -o pipefail

programname=`basename "$0"`

display_usage() {
    echo "usage: $programname (create|restore) source destination"
    echo "  create         create snapshot file from docker volume"
    echo "  restore        restore snapshot file to docker volume"
    echo "  source         source path"
    echo "  destination    destination path"
    echo
    echo "Tip: Supports tar's compression algorithms automatically"
    echo "     based on the file extention, for example .tar.gz"
    echo
    echo "Examples:"
    echo "docker-volume-snapshot create xyz_volume xyz_volume.tar"
    echo "docker-volume-snapshot create xyz_volume xyz_volume.tar.gz"
    echo "docker-volume-snapshot restore xyz_volume.tar xyz_volume"
    echo "docker-volume-snapshot restore xyz_volume.tar.gz xyz_volume"
}

case "$1" in
    "create")
        if [[ -z "$2" || -z "$3" ]]; then display_usage; exit 1; fi
        directory=`dirname "$3"`
        if [ "$directory" == "." ]; then directory=$(pwd); fi
        filename=`basename "$3"`
        docker run --rm -v "$2:/source" -v "$directory:/dest" busybox tar cvaf "/dest/$filename" -C /source .
        ;;
    "restore")
        if [[ -z "$2" || -z "$3" ]]; then display_usage; exit 1; fi
        directory=`dirname "$2"`
        if [ "$directory" == "." ]; then directory=$(pwd); fi
        filename=`basename "$2"`
        docker run --rm -v "$3:/dest" -v "$directory:/source" busybox tar xvf "/source/$filename" -C /dest
        ;;
    *)
        display_usage
        exit 1 # Command to come out of the program with status 1
        ;;
esac
