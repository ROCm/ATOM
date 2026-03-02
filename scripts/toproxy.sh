


# curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{"prompt": "the us is ?","max_tokens": 10,"temperature": 0, "top_k":1}' | awk -F'"' '{print $22}'
curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{"prompt": "1 2 3 4 5 ","max_tokens": 10,"temperature": 0, "top_k":1}' | awk -F'"' '{print $22}'


# curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{"prompt": "the the the the the the the the the the","max_tokens": 10,"temperature": 0, "top_k":1}' | awk -F'"' '{print $22}'

# curl -X POST -s http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{"prompt": "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ","max_tokens": 10,"temperature": 0, "top_k":1}' | awk -F'"' '{print $22}'

# curl -X POST -s http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d '{"prompt": "1 2 3 4 5 ","max_tokens": 10,"temperature": 0, "top_k":1}' | awk -F'"' '{print $22}'


# curl -X POST -s http://127.0.0.1:8000/v1/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "prompt": "1 2 3 4 5 ",
#     "max_tokens": 10,
#     "temperature": 0, 
#     "top_k": 1,
#     "kv_transfer_params": {
#       "do_remote_decode": true,
#       "do_remote_prefill": false
#     }
#   }' | awk -F'"' '{print $22}'


  # curl -X POST -s http://127.0.0.1:10001/v1/completions \
  # -H "Content-Type: application/json" \
  # -d '{
  #   "prompt": "1 2 3 4 5 ",
  #   "max_tokens": 10,
  #   "temperature": 0, 
  #   "top_k": 1,
  # }' 







# curl -X POST -s http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d '{"prompt": "1 2 3 4 5 ","max_tokens": 10,"temperature": 0, "top_k":1}' | awk -F'"' '{print $22}'