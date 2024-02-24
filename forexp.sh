make clean
make
SET=$(seq 0 9)
for i in $SET
do
    ./run.sh model.bin ./test1/re.txt 1048576 1621
done
