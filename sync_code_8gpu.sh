sync_code() {
  rsync -avP --exclude-from=$HOME/github/ASDF/$1/.gitignore --exclude='.git' --checksum --delete $HOME/github/ASDF/$1/ 8gpu:/root/github/$1/
}

rsync -avP --checksum $HOME/chLi/Dataset/Transformers 8gpu:/root/chLi/Dataset/

sync_code mash-diffusion
sync_code base-trainer

sync_code mash-occ-decoder

sync_code ma-sh
sync_code distribution-manage
sync_code sdf-generate
sync_code open3d-manage
sync_code param-gauss-recon
sync_code open-clip-detect
sync_code dino-v2-detect
sync_code ulip-manage
sync_code wn-nc

sync_code CFM
sync_code light-field-distance
