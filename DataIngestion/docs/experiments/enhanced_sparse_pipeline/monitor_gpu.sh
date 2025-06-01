#!/bin/bash

# Real-time GPU Monitoring for Enhanced Sparse Pipeline
# Shows GPU utilization, memory, and performance metrics during benchmarks

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}GPU Performance Monitor${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. Please install NVIDIA drivers.${NC}"
    exit 1
fi

# Create monitoring log
MONITOR_LOG="gpu_monitor_$(date +%Y%m%d_%H%M%S).log"

# Function to display GPU stats
show_gpu_stats() {
    clear
    echo -e "${BLUE}=== Enhanced Sparse Pipeline - GPU Monitor ===${NC}"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Log: $MONITOR_LOG"
    echo ""
    
    # Get GPU info
    gpu_info=$(nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader)
    echo -e "${GREEN}GPU Info:${NC} $gpu_info"
    echo ""
    
    # Get current stats
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    while IFS=',' read -r gpu_util mem_util mem_used mem_total temp power; do
        # Trim whitespace
        gpu_util=$(echo $gpu_util | xargs)
        mem_util=$(echo $mem_util | xargs)
        mem_used=$(echo $mem_used | xargs)
        mem_total=$(echo $mem_total | xargs)
        temp=$(echo $temp | xargs)
        power=$(echo $power | xargs)
        
        # Calculate memory percentage
        mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc)
        
        # GPU Utilization with color coding
        echo -n "GPU Utilization: "
        if [ "$gpu_util" -ge 85 ]; then
            echo -e "${GREEN}${gpu_util}%${NC} ✓ (Target: 85-95%)"
        elif [ "$gpu_util" -ge 70 ]; then
            echo -e "${YELLOW}${gpu_util}%${NC} (Below target)"
        else
            echo -e "${RED}${gpu_util}%${NC} (Low utilization)"
        fi
        
        # Memory Usage
        echo "Memory Usage: ${mem_used} MB / ${mem_total} MB (${mem_percent}%)"
        
        # Temperature
        echo -n "Temperature: "
        if [ "$temp" -le 75 ]; then
            echo -e "${GREEN}${temp}°C${NC}"
        elif [ "$temp" -le 85 ]; then
            echo -e "${YELLOW}${temp}°C${NC}"
        else
            echo -e "${RED}${temp}°C${NC} (High!)"
        fi
        
        # Power Draw
        echo "Power Draw: ${power}W"
        
        # Visual bars
        echo ""
        echo -n "GPU: ["
        gpu_bar_len=$((gpu_util * 50 / 100))
        for i in $(seq 1 50); do
            if [ $i -le $gpu_bar_len ]; then
                if [ "$gpu_util" -ge 85 ]; then
                    echo -ne "${GREEN}█${NC}"
                elif [ "$gpu_util" -ge 70 ]; then
                    echo -ne "${YELLOW}█${NC}"
                else
                    echo -ne "${RED}█${NC}"
                fi
            else
                echo -n "░"
            fi
        done
        echo "] ${gpu_util}%"
        
        echo -n "MEM: ["
        mem_bar_len=$(echo "scale=0; $mem_percent * 50 / 100" | bc)
        for i in $(seq 1 50); do
            if [ $i -le $mem_bar_len ]; then
                echo -ne "${BLUE}█${NC}"
            else
                echo -n "░"
            fi
        done
        echo "] ${mem_percent}%"
        
        # Log to file
        echo "$(date '+%Y-%m-%d %H:%M:%S'),$gpu_util,$mem_used,$mem_total,$temp,$power" >> "$MONITOR_LOG"
    done
    
    # Show running processes
    echo ""
    echo -e "${BLUE}GPU Processes:${NC}"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader | head -5
    
    # Performance hints
    echo ""
    echo -e "${BLUE}Performance Hints:${NC}"
    if [ "$gpu_util" -lt 85 ] 2>/dev/null; then
        echo "• Consider increasing batch size for better GPU utilization"
        echo "• Check if enhanced mode is enabled (ENHANCED_MODE=true)"
    fi
    if [ $(echo "$mem_percent > 90" | bc) -eq 1 ] 2>/dev/null; then
        echo "• Memory usage high - consider reducing batch size"
    fi
}

# Header for log file
echo "timestamp,gpu_util,mem_used_mb,mem_total_mb,temp_c,power_w" > "$MONITOR_LOG"

# Main monitoring loop
echo "Starting GPU monitoring..."
echo "Press Ctrl+C to stop"
echo ""

# Trap Ctrl+C to show summary
trap 'show_summary' INT

show_summary() {
    echo -e "\n\n${BLUE}=== Monitoring Summary ===${NC}"
    
    if [ -f "$MONITOR_LOG" ]; then
        # Calculate averages
        avg_gpu=$(awk -F',' 'NR>1 {sum+=$2; count++} END {if(count>0) print sum/count; else print 0}' "$MONITOR_LOG")
        max_gpu=$(awk -F',' 'NR>1 {if($2>max) max=$2} END {print max}' "$MONITOR_LOG")
        avg_mem=$(awk -F',' 'NR>1 {sum+=$3; count++} END {if(count>0) print sum/count; else print 0}' "$MONITOR_LOG")
        max_mem=$(awk -F',' 'NR>1 {if($3>max) max=$3} END {print max}' "$MONITOR_LOG")
        
        echo "Average GPU Utilization: ${avg_gpu}%"
        echo "Peak GPU Utilization: ${max_gpu}%"
        echo "Average Memory Usage: ${avg_mem} MB"
        echo "Peak Memory Usage: ${max_mem} MB"
        echo ""
        echo "Log saved to: $MONITOR_LOG"
    fi
    
    exit 0
}

# Monitor loop
while true; do
    show_gpu_stats
    sleep 1
done