git pull

if [ -d "../sdf-generate/" ]; then
  cd ../sdf-generate/
  git pull
fi

if [ -d "../open3d-manage/" ]; then
  cd ../open3d-manage/
  git pull
fi

if [ -d "../param-gauss-recon/" ]; then
  cd ../param-gauss-recon/
  git pull
fi

if [ -d "../open-clip-detect/" ]; then
  cd ../open-clip-detect/
  git pull
fi

if [ -d "../dino-v2-detect/" ]; then
  cd ../dino-v2-detect/
  git pull
fi

if [ -d "../ulip-manage/" ]; then
  cd ../ulip-manage/
  git pull
fi

if [ -d "../distribution-manage/" ]; then
  cd ../distribution-manage/
  git pull
fi

if [ -d "../chamfer-distance/" ]; then
  cd ../chamfer-distance/
  git pull
fi
