# common.sh: control-host settings, sourced by the ctl/ scripts (README section 8). The control host reaches both
# trays over ssh (users / keys in ~/.ssh/config), or is one of them; the kit (with its site.env) sits at $KIT_REMOTE
# on both trays.
P_HOST=${P_HOST:?set P_HOST (ssh name of the prefill tray)}
D_HOST=${D_HOST:?set D_HOST (ssh name of the decode tray)}
KIT_REMOTE=${KIT_REMOTE:?set KIT_REMOTE (the kit path on both trays)}
CTR=${PD_CONTAINER:?set PD_CONTAINER (the container name in the kit site.env)}
on() {  # on <host> <command> [timeout s]: runs here when <host> is this machine, else over ssh
  if [ "${1%%.*}" = "$(hostname -s)" ]; then
    timeout "${3:-120}" bash -c "$2"
  else
    timeout "${3:-120}" ssh -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR "$1" "$2"
  fi
}
