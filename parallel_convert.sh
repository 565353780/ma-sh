SCRIPT_PATH=$1
PROCESSOR_NUM=$2

for i in $(seq 1 ${PROCESSOR_NUM}); do
  python ${SCRIPT_PATH} &
  sleep 1
  echo "started Convertor No."$i
done

wait

echo "all Convertors finished!"
