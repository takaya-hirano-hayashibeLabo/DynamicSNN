for TRIAL in {1..100}
do 
    echo "[${TRIAL}/100]====================================================================="
    python exp1.1.py --trial ${TRIAL} 
done