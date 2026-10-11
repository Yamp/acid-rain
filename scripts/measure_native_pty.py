#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Measure uninstrumented protected-binary PTY frame cadence, without an emulator."""
import argparse,errno,fcntl,hashlib,importlib.util,json,os,select,statistics,struct,sys,termios,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary',type=Path,default=ROOT/'benchmarks/whirlpool-after/acid-rain')
parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/native-pty')
parser.add_argument('--frames',type=int,default=450)
parser.add_argument('--warmup',type=int,default=90)
parser.add_argument('--sizes',default='200x60,160x50,320x100')
parser.add_argument('--blending',choices=('baseline','on','off'),default='baseline')
parser.add_argument('--seconds',type=float,default=0.0,help='Use a fixed wall-time measurement window instead of a fixed display-frame count')
parser.add_argument('--warmup-seconds',type=float,default=3.0)
parser.add_argument('--water',choices=('classic','rapids','drain'),default='drain')
parser.add_argument('--quality',choices=('auto','high','balanced','low'),default='auto')
parser.add_argument('--speed',type=float,default=3.0)
parser.add_argument('--intensity',type=float,default=2.0)
parser.add_argument('--bleach',type=float,default=1.0)
parser.add_argument('--rain',type=float,default=6.0)
parser.add_argument('--drop-size',type=float,default=1.0)
parser.add_argument('--rgb-shift',type=float,default=None)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);binary=args.binary.resolve()
if not (.1<=args.speed<=3 and 0<=args.intensity<=4 and 0<=args.bleach<=4 and 0<=args.rain<=60 and .25<=args.drop_size<=3 and (args.rgb_shift is None or 0<=args.rgb_shift<=12)):
 parser.error('invalid water/optics controls')
assert args.frames>args.warmup+1
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();proof=json.loads((binary.parent/'provenance.json').read_text());assert sha(binary)==proof['rebuilt_native_binary_sha256']
spec=importlib.util.spec_from_file_location('pty_smoke',ROOT/'scripts/pty_smoke.py');pty=importlib.util.module_from_spec(spec);spec.loader.exec_module(pty)
settings={'water':args.water,'quality':args.quality,'speed':args.speed,'intensity':args.intensity,'bleach':args.bleach,'rain':args.rain,'drop_size':args.drop_size,'rgb_shift':args.rgb_shift,'rgb_effects':'on (verified help)','seed':7,'scene':'journey'}
metadata={'binary':str(binary),'binary_sha256':sha(binary),'build_provenance':proof,'frames':args.frames,'warmup':args.warmup,'blending':args.blending,'measurement_seconds':args.seconds,'warmup_seconds':args.warmup_seconds,'requested_controls':settings,'clock':'display/output sync markers','simulation_fps':None,'source_render_fps':None,'note':'Uninstrumented protected binary. Fast PTY reader; sync markers timestamped per read. Intermediate frames count as display output, not new simulation or source-render frames. Simulation/render frequency is not inferred from this trace. No native terminal emulator is present. Timed frames precede q; exact terminal restoration is verified.','cases':[]}
for w,h in [tuple(map(int,size.split('x'))) for size in args.sizes.split(',')]:
 extra=('--water',args.water,'--scene','journey','--speed',str(args.speed),'--intensity',str(args.intensity),'--bleach',str(args.bleach),'--rain',str(args.rain),'--drop-size',str(args.drop_size))
 if args.rgb_shift is not None:extra+=('--rgb-shift',str(args.rgb_shift))
 t=pty.TerminalProcess(binary,w,h,extra,quality=args.quality);events=[];tail=b'';quit_sent=False;captured=0;controls=[];runtime_error=False
 try:
  t.wait_frames(2)
  offset=len(t.data);t.send(b'h');t.wait_for(lambda:b'ACID RAIN / CONTROLS' in t.data[offset:],'benchmark settings help')
  screen=bytes(t.data[offset:]);screen=screen[screen.index(b'ACID RAIN / CONTROLS'):]
  expected_settings=[f'Water: {args.water.title()}',f'Quality: {args.quality.title()}',f'Speed: {args.speed:.2f}x',f'Trip: {args.intensity:.1f}',f'Rain: {args.rain:.1f}/s',f'Drop size: {args.drop_size:.2f}x',f'Bleach: {args.bleach:.2f}x','RGB: on']
  if args.rgb_shift is not None:expected_settings.append(f'RGB shift: {args.rgb_shift:.2f}px')
  for expected in expected_settings:assert expected.encode() in screen,f'benchmark setting missing from help: {expected}'
  t.send(b'h');t.wait_frames(2)
  for state in (() if args.blending=='baseline' else ('off',) if args.blending=='off' else ('off','on')):
   offset=len(t.data);sent=time.monotonic();t.send(b'i');expected=f'Frame blending: {state} | i: toggle'.encode()
   t.wait_for(lambda:expected in t.data[offset:],f'blending {state} HUD',timeout=1.0)
   controls.append({'key':'i','state':state,'latency_ms':(time.monotonic()-sent)*1000.0,'verified_hud':expected.decode()})
  t.drain(.05);base_frames=t.frames;start=time.monotonic();start_ns=time.monotonic_ns();deadline=start+max(120,args.seconds+args.warmup_seconds+10)
  while True:
   if time.monotonic()>deadline:raise TimeoutError(f'{w}x{h}')
   ready=select.select([t.master],[],[],.02)[0]
   if ready:
    try:chunk=os.read(t.master,262144)
    except BlockingIOError:continue
    except OSError as e:
     if e.errno==errno.EIO:break
     raise
    stamp=time.monotonic_ns()
    if chunk:
     combined=tail+chunk;offset=len(tail)
     for marker,kind in [(pty.SYNC_BEGIN,'begin'),(pty.SYNC_END,'end')]:
      pos=0
      while (pos:=combined.find(marker,pos))>=0:
       if pos+len(marker)>offset:events.append((stamp,kind))
       pos+=len(marker)
     runtime_error |= any(marker in combined.lower() for marker in (b'panicked at',b'fatal runtime error'))
     tail=combined[-32:];captured+=len(chunk);t.data.extend(chunk)
     if len(t.data)>32768:del t.data[:-32768]
     t.frames+=(t.marker_tail+chunk).count(pty.SYNC_BEGIN);t.marker_tail=(t.marker_tail+chunk)[-(len(pty.SYNC_BEGIN)-1):]
   finished=(time.monotonic()-start>=args.warmup_seconds+args.seconds) if args.seconds>0 else t.frames-base_frames>=args.frames
   if finished and not quit_sent:t.send(b'q');quit_sent=True
   if t.poll() is not None and not ready:break
  assert t.exit_code==0
  if sys.platform=='darwin':
   # ICANON restoration sets transient PENDIN. Settle the line discipline
   # without consuming input or masking any saved terminal attribute.
   fcntl.ioctl(t.slave,termios.FIONREAD,struct.pack('I',0))
  assert termios.tcgetattr(t.slave)==t.original_termios
  assert not runtime_error
 finally:t.close()
 if args.seconds>0:
  lower=start_ns+int(args.warmup_seconds*1e9);upper=lower+int(args.seconds*1e9)
  begins=[n for n,kind in events if kind=='begin' and lower<=n<upper]
 else:begins=[n for n,kind in events if kind=='begin'][:args.frames][args.warmup:]
 assert len(begins)>1,'too few displayed frames for a cadence estimate'
 delta=sorted((b-a)/1e6 for a,b in zip(begins,begins[1:]));mean=statistics.mean(delta)
 result={'size':[w,h],'argv':[str(binary),'--fps','60','--quality',args.quality,'--seed','7',*extra],'blending':args.blending,'controls':controls,'verified_help_settings':expected_settings,'requested_controls':settings,'interval_ms':{'mean':mean,'p50':delta[(len(delta)-1)*50//100],'p95':delta[(len(delta)-1)*95//100],'max':max(delta)},'display_fps':1000/mean,'simulation_fps':None,'source_render_fps':None,'timed_intervals':len(delta),'captured_display_frames':t.frames-base_frames,'captured_bytes':captured,'exit_code':t.exit_code,'terminal_restored':True,'started_monotonic_ns':start_ns,'elapsed_seconds':time.monotonic()-start}
 name=f'{args.water}-{w}x{h}-{args.quality}-speed{args.speed:g}';(args.output/(name+'.timestamps.csv')).write_text('monotonic_ns,event\n'+''.join(f'{stamp},{kind}\n' for stamp,kind in events))
 metadata['cases'].append(result);(args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(result),flush=True)
assert sha(binary)==proof['rebuilt_native_binary_sha256']
lines=['# Uninstrumented protected-binary PTY display cadence','','| Size | Blending | Mean interval ms | p50 ms | p95 ms | Max ms | Display frames/s |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for r in metadata['cases']:
 x=r['interval_ms'];lines.append(f"| {r['size'][0]}×{r['size'][1]} | {r['blending']} | {x['mean']:.3f} | {x['p50']:.3f} | {x['p95']:.3f} | {x['max']:.3f} | {r['display_fps']:.2f} |")
window=f'{args.warmup_seconds:g}s warmup plus {args.seconds:g}s measured wall time' if args.seconds>0 else f'First {args.warmup} of {args.frames} display frames excluded as warmup'
lines+=['',f'Fast PTY reader; no native terminal emulator. Explicit settings: {json.dumps(settings,sort_keys=True)}. {window}.','Display cadence counts intermediate blended frames; source-render and simulation frequencies are not measured by this trace. Blending on/off is selected through the actual i key with a verified HUD, not startup flags.']
(args.output/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
