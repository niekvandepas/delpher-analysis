# If you come from bash you might have to change your $PATH.
# export PATH=$HOME/bin:/usr/local/bin:$PATH

# Path to your oh-my-zsh installation.
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"

zstyle ':omz:update' mode auto      # update automatically without asking
plugins=(z)

source $ZSH/oh-my-zsh.sh

alias ga='git add'
alias gs='git status'
alias GS='git status'
alias gc='git commit'
alias gr='git reset'
alias gl='git log'
alias gaa="git add -A"
alias gb='git --no-pager branch'
alias gch='git checkout'
alias gsh='git show'
alias gd='git diff'
alias gp='git push'
alias gm='git merge'
alias grup='git remote update'
alias glm='git log --author niek'
alias gap='git add --patch'
alias gdc='git diff --cached'
alias gsp='git stash --patch'
alias gcam='git commit --amend'
alias gctemp='git commit --no-verify -m temp'
alias gcwip='git commit --no-verify -m wip'
alias gcnow='git add -A && git commit -m "`date`"'
alias gcane='git commit --amend --no-edit'
alias gcn='git commit --no-verify'
alias grs='git reset --soft HEAD~'
alias gdt='git difftool'
alias gcstyle='git commit -m "style"'
alias gcchore='git commit -nm "chore"'
alias gcdoc='git commit -m "doc"'
alias gpl='git pull'
alias gshw='git show --word-diff'

alias rgrep='grep -r'
alias igrep='grep -i'
alias rigrep='grep -ri'
alias eg='egrep'
alias up='cd ..'
alias back='cd $OLDPWD'
alias c='code --disable-workspace-trust'
alias C='code --disable-workspace-trust'
alias e='eza'
alias m='make'
alias o='open'
alias t='trash'
alias x=exit
alias X=exit
alias l=less
# alias less='bat'
alias cb='cabal build'
alias ct='cabal test'
alias cre='cabal repl'
alias cra='cargo run'
alias car='cargo run'
alias carr='cargo run --release'
alias cab='cargo build'
alias cart='cargo test'
alias ns='npm start'
alias copy=pbcopy
alias cop=pbcopy
alias youtubedl='yt-dlp'
alias compile='./compile.sh'
alias au='./compile.sh'
alias hn=hostname
alias gcm='generate_commit_message'
alias sizes='du -hd 1 | sort -hr'
alias j='just'
alias v='nvim'
alias n='nvim'
alias f='fuck'
alias vac='source ./venv/bin/activate'
