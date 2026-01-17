#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <log_file>"
    exit 1
fi

LOG_FILE="$1"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: File '$LOG_FILE' not found."
    exit 1
fi

echo "=== TP0_EP0 ==="
grep prefill_attention_asm -i "$LOG_FILE"  | grep "pid= " -i | grep TP0_EP0 -i
echo "=== TP1_EP1 ==="
grep prefill_attention_asm -i "$LOG_FILE"  | grep "pid= " -i | grep TP1_EP1 -i
echo "=== TP2_EP2 ==="
grep prefill_attention_asm -i "$LOG_FILE"  | grep "pid= " -i | grep TP2_EP2 -i
echo "=== TP3_EP3 ==="
grep prefill_attention_asm -i "$LOG_FILE"  | grep "pid= " -i | grep TP3_EP3 -i
echo "=== TP4_EP4 ==="
grep prefill_attention_asm -i "$LOG_FILE"  | grep "pid= " -i | grep TP4_EP4 -i
echo "=== TP5_EP5 ==="
grep prefill_attention_asm -i "$LOG_FILE"  | grep "pid= " -i | grep TP5_EP5 -i
echo "=== TP6_EP6 ==="
grep prefill_attention_asm -i "$LOG_FILE"  | grep "pid= " -i | grep TP6_EP6 -i
echo "=== TP7_EP7 ==="
grep prefill_attention_asm -i "$LOG_FILE"  | grep "pid= " -i | grep TP7_EP7 -i
