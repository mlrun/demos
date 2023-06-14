#!/usr/bin/env bash

set -o errexit
set -o pipefail

SCRIPT="$(basename "$0")"

git_owner=mlrun
git_repo=demos
git_base_url="https://github.com/${git_owner}/${git_repo}" 
git_url="${git_base_url}.git"
user=${V3IO_USERNAME}

USAGE="\
$SCRIPT:
Retrieves updated demos from the mlrun/demos GitHub repository.
USAGE: ${SCRIPT} [OPTIONS]
OPTIONS:
  -h|--help   -  Display this message and exit.
  -b|--branch -  Git branch name. Default: The latest release branch that
                 matches the version of the installed 'mlrun' package.
  -u|--user   -  Username, which determines the directory to which to copy the
                 retrieved demo files (/v3io/users/<username>).
                 Default: \$V3IO_USERNAME, if set to a non-empty string.
  --mlrun-ver -  The MLRun version for which to get demos; determines the Git
                 branch from which to get the demos, unless -b|--branch is set.
                 Default: The version of the installed 'mlrun' package.
  --dry-run   -  Show files to update but don't execute the update.
  --no-backup -  Don't back up the existing demos directory before the update.
                 Default: Back up the existing demos directory to a
                 /v3io/users/<username>/demos.old/<timestamp>/ directory."

error_exit()
{

# ----------------------------------------------------------------
# Function for exit due to fatal program error
#   Accepts 1 argument:
#     string containing descriptive error message
# ----------------------------------------------------------------
  echo "${SCRIPT}: ${1:-"Unknown Error"}" 1>&2
  exit 1
}

error_usage()
{
    echo "${SCRIPT}: ${1:-"Unknown Error"}" 1>&2
    echo -e "$USAGE"
    exit 1
}

while :
do
    case $1 in
        -h | --help) echo -e "$USAGE"; exit 0 ;;
        -b|--branch)
            if [ "$2" ]; then
                branch=$2
                shift
            else
                error_usage "$1: Missing branch name."
            fi
            ;;
        --branch=?*)
            branch=${1#*=} # Delete everything up to "=" and assign the remainder.
            ;;
        --branch=)         # Handle the case of an empty --branch=
            error_usage "$1: Missing branch name."
            ;;
        -u|--user)
            if [ "$2" ]; then
                user=$2
                shift
            else
                error_usage "$1: Missing username."
            fi
            ;;
        --user=?*)
            user=${1#*=} # Delete everything up to "=" and assign the remainder.
            ;;
        --user=)         # Handle the case of an empty --user=
            error_usage "$1: Missing username."
            ;;
        --mlrun-ver)
            if [ "$2" ]; then
                mlrun_version=$2
                shift
            else
                error_usage "$1: Missing MLRun version."
            fi
            ;;
        --mlrun-ver=?*)
            mlrun_version=${1#*=} # Delete everything up to "=" and assign the remainder.
            ;;
        --mlrun-ver=)         # Handle the case of an empty --mlrun-ver=
            error_usage "$1: Missing MLRun version."
            ;;
        --dry-run)
            dry_run=1
            ;;
        --no-backup)
            no_backup=1
            ;;
        -*) error_usage "$1: Unknown option."
            ;;
        *) break;
    esac
    shift
done

if [ -z "${user}" ]; then
    error_usage "Missing username."
fi

# shellcheck disable=SC2236
if [ ! -z "${dry_run}" ]; then
    echo "Dry run; no files will be copied."
fi

# shellcheck disable=SC2236
if [ ! -z "${no_backup}" ]; then
    echo "The existing demos directory won't be backed up before the update."
fi

if [ -z "${branch}" ]; then

    if [ -z "${mlrun_version}" ]; then
        pip_mlrun=$(pip show mlrun | grep Version) || :

        if [ -z "${pip_mlrun}" ]; then
            error_exit "MLRun version not found. Aborting..."
        else
            echo "Detected MLRun version: ${pip_mlrun}"
            mlrun_version="${pip_mlrun##Version: }"
        fi
    else
        echo "Looking for demos for the specified MLRun version - ${mlrun_version}."
    fi
    
    # shellcheck disable=SC2006
    tag_prefix=`echo "${mlrun_version}" | cut -d . -f1-2`
    # shellcheck disable=SC2006
    latest_tag=`git ls-remote --tags --refs --sort=-v:refname ${git_base_url} | grep "${mlrun_version%%r*}" | grep -v '\^{}' | grep -v 'rc' | grep -v 'RC' | head -n1 | awk '{print $2}' | sed 's#refs/tags/##'`
    if [ -z "${latest_tag}" ]; then
        error_exit "Couldn't locate a Git tag with prefix 'v${tag_prefix}.*'."
        # shellcheck disable=SC2006
        latest_tag=`git ls-remote --tags --refs --sort=-v:refname ${git_base_url} | grep "${mlrun_version%%r*}" | grep -v '\^{}' | head -n1 | awk '{print $2}' | sed 's#refs/tags/##'`
    else
        # Remove the prefix from the Git tag
        branch=${latest_tag#refs/tags/}
        echo "Detected ${git_url} tag: ${branch}"
    fi
fi

dest_dir="/v3io/users/${user}"
demos_dir="${dest_dir}/demos"
echo "Updating demos from ${git_url} branch ${branch} to '${demos_dir}'..."

temp_dir=$(mktemp -d /tmp/temp-get-demos.XXXXXXXXXX)
trap '{ rm -rf $temp_dir; }' EXIT
echo "Copying files to a temporary directory '${temp_dir}'..."

tar_url="${git_base_url}/archive/${branch}.tar.gz"
wget -qO- "${tar_url}" | tar xz -C "${temp_dir}" --strip-components 1

if [ -z "${dry_run}" ]; then
    if [ -d "${demos_dir}" ]; then

        if [ -z "${no_backup}" ]; then
            # Back up the existing demos directory
            dt=$(date '+%Y%m%d%H%M%S');
            old_demos_dir="${dest_dir}/demos.old/${dt}"

            echo "Moving existing '${demos_dir}' to ${old_demos_dir}'..."

            mkdir -p "${old_demos_dir}"
            cp -r "${demos_dir}/." "${old_demos_dir}" && rm -rf "${demos_dir}"
        else
            rm -rf "${demos_dir}"
        fi
    fi

    echo "Copying files to '${demos_dir}'..."
    mkdir -p "${demos_dir}"
    cp -RT "${temp_dir}" "${demos_dir}"
else
    # Dry run
    echo "Identified the following files to copy to '${dest_dir}':"
    find "${temp_dir}/" -not -path '*/\.*' -type f -printf "%p\n" | sed -e "s|^${temp_dir}/|./demos/|"
fi

echo "Deleting temporary directory '${temp_dir}..."
rm -rf "${temp_dir}"

echo "DONE"