git pull

if [ -d "../chamfer-distance/" ]; then
  cd ../chamfer-distance/
  git pull
fi

if [ -d "../mesh-graph-cut/" ]; then
  cd ../mesh-graph-cut/
  git pull
fi

if [ -d "../diff-curvature/" ]; then
  cd ../diff-curvature/
  git pull
fi

if [ -d "../nvdiffrast/" ]; then
  cd ../nvdiffrast/
  git pull
fi

if [ -d "../pytorch3d/" ]; then
  cd ../pytorch3d/
  git pull
fi
