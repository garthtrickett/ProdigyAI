python depth_snapshot_list.py
foo="`cat orderbook_symbol_string.txt`"

IFS=',' # space is set as delimiter
read -ra ADDR <<< "$foo" # str is read into an array as tokens separated by IFS
for i in "${ADDR[@]}"; do # access each element of array
    echo "$i"
    logfile="${i}_orderbook_thirdparty_code.log"
    python orderbook_thirdparty_code.py --symbol $i > $logfile &
done