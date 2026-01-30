# alias caa='cd /home/zejchen/plugin/ATOM/test_plugin'
# alias cvv='cd /home/zejchen/rocm_vllm/vllm'
# alias css='cd /home/zejchen/zejun_sglang/sglang'
alias caa='cd /home/zejchen@amd.com/plugin/ATOM/test_plugin'

alias gg='git fetch && git checkout origin/zejun/plugin_for_atom_1223'
alias ggg='git fetch && git checkout origin/zejun/model_impl'
# alias gb='git checkout 1ce085879dd473dae397cabbf57a4a3240530a61'
alias gbb='git checkout origin/zejun/plugin_for_atom_1223'

alias tt0='bash ./test.0.6B.tp1.no.custom.sh'
alias tt1='bash ./test.0.6B.tp1.sh'
# alias tt2='bash ./test.235B.tp4.sh'
alias tt3='bash ./test.235B.tp8.ep8.sh'
alias tt4='bash ./test.235B.tp8.ep8.original.sh'

alias cc='bash ./curl.sh'
# alias kk='rm -rf ./*.pt'

# kill all
#             ps -eo pid,comm | grep -E '^( *[0-9]+ +(python3|tee|VLLM::))' | awk '{print $1}' | xargs kill -9

pip install --upgrade triton
pip install hf_transfer
