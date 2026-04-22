#!/bin/bash
API="https://stemscriber.com/api"
upload_song() {
  local path="$1"
  local label="$2"
  local resp=$(curl -s -X POST "$API/upload" \
    -F "file=@$path" \
    -F "gp_tabs=false" \
    -F "chord_detection=true" \
    -F "plan=free")
  local job_id=$(echo "$resp" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('job_id','ERR:'+str(d)))")
  echo "$label $job_id"
}
upload_song "/Users/jeffkozelski/Desktop/Jamiroquai - The Complete Discography/[Albums]/(1996) Jamiroquai - Travelling Without Moving/05 - Alright.mp3" "ALRIGHT"
upload_song "/Users/jeffkozelski/Desktop/Jamiroquai - The Complete Discography/[Albums]/(1996) Jamiroquai - Travelling Without Moving/02 - Cosmic Girl.mp3" "COSMIC"
upload_song "/Users/jeffkozelski/Desktop/Steely Dan, Donald Fagen, Walter Becker - Discography (1972-2012)/1 Steely Dan/#6 Aja (1976)/01 - Black Cow.mp3" "BLACKCOW"
upload_song "/Users/jeffkozelski/Desktop/Steely Dan, Donald Fagen, Walter Becker - Discography (1972-2012)/1 Steely Dan/#6 Aja (1976)/04 - Peg.mp3" "PEG"
