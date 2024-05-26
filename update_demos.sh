#!/usr/bin/env bash

set -o errexit
set -o pipefail

SCRIPT="$(basename "$0")"

git_owner=mlrun
git_repo=mlrun
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
                 /v3io/users/<username>/demos.old/<timestamp>/ directory.
  --path      -  Demos folder download path."

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

get_latest_tag() {
    local mlrun_version="$1"
    local git_owner="$2"
    local git_repo="$3"
    local git_base_url="$4" # Unused in this function but can be useful for future enhancements
    local git_url="$5"

    # Fetch tags from git
    local tags=($(git ls-remote --tags --refs --sort='v:refname' "${git_url}" | awk '{print $2}'))
    # Initialize two empty arrays to hold the two separate lists
    with_rc=()
    without_rc=()
    # Iterate through the list of version strings to split between latest and release
    for version in "${tags[@]}"; do
      tag=${version#refs/tags/}
      if [[ $version == *"rc"* ]]; then
        # If the version string contains "rc," add it to the list with "rc" - only the ones in the form of "something"rcXX
        if [[ $version =~ (^|[^[:alnum:]])rc[0-9]{1,2}$ ]]; then
            with_rc+=("$tag")
        fi
      else
        # Otherwise, add it to the list without "rc" 
        without_rc+=("$tag")
      fi
    done
    
    formatted_version=$(echo "$mlrun_version" | sed -E 's/.*([0-9]+\.[0-9]+\.[0-9]+).*$/\1/')
    # finding whether there is a release
    for item in "${without_rc[@]}"; do
      if [[ $item == *"$formatted_version"* ]]; then
        echo "$item"
        return
      fi
    done
    
    # if release doesn't exists, find matching rc
    formatted_rc=$(echo "$mlrun_version" | sed -E 's/.*rc([0-9]+)?.*/-rc\1/')
    if [ "$formatted_rc" == "$mlrun_version" ]; then # couldn't find rc (mlrun_version is a release with no rc)
      formatted_rc=""
    fi
    
    all_rcs=()
    for item in "${with_rc[@]}"; do
      if [[ $item == *"$formatted_version"* ]]; then
        all_rcs+=("$item")
      fi
    done
    
    if [ -z "$all_rcs" ]; then # couldn't find any version, returning latest release
      echo "${without_rc[@]}" | tr ' ' '\n' | sort -r | head -n 1
      return
    else
      # trying to find matching rc
      # case mlrun doesnt have an rc (its a release) and demos doesn't have matching release (fetching latest rc)
      if [ -z "$formatted_rc" ]; then # rc is ""
        echo "${with_rc[*]}" | tr ' ' '\n' | sort -Vr | head -n 1
        return
      fi
      # case mlrun does have an rc - return its matching demos rc
      for item in "${all_rcs[@]}"; do
        if [[ $item == *"$formatted_rc"* ]]; then
          echo "$item"
          return
        fi
      done
      # coldn't find matching rc (mlrun does have an rc but demos doesn't have a matching one) returns latest rc
      echo "${with_rc[*]}" | tr ' ' '\n' | sort -Vr | head -n 1
      return
    fi 
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
        --path=?*)
            demos_dir=${1#*=} # Delete everything up to "=" and assign the remainder.
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
    latest_tag=$(get_latest_tag "${mlrun_version}" "${git_owner}" "${git_repo}" "${git_base_url}" "${git_url}")
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

# If --path argument is specified
if [ -z "${demos_dir}" ]; then
    dest_dir="/v3io/users/${user}"
    demos_dir="${dest_dir}/demos"
fi

echo "Updating demos from ${git_url} branch ${branch} to '${demos_dir}'..."

temp_dir=$(mktemp -d /tmp/temp-get-demos.XXXXXXXXXX)
trap '{ rm -rf $temp_dir; }' EXIT
echo "Copying files to a temporary directory '${temp_dir}'..."

if [[ "${branch}">"v1.7" ]]; then
    tar_url="https://github.com/mlrun/mlrun/releases/download/${branch}/mlrun-demos.tar"
    echo "Downloading : $tar_url ..."
    wget "${tar_url}"
    tar -xvf mlrun-demos.tar -C "${temp_dir}" --strip-components 1
else
    tar_url="https://github.com/mlrun/demos/archive/${branch}.tar.gz"
    echo "Downloading : $tar_url ..."
    wget -qO- "${tar_url}" | tar xz -C "${temp_dir}" --strip-components 1
fi

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
