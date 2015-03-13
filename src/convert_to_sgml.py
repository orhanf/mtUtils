import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--src-lang", type=str, default='')
parser.add_argument("--trg-lang", type=str, default='')
parser.add_argument("--setid", type=str, default='')
parser.add_argument("--src", action="store_true", default=False)
args = parser.parse_args()

def main_loop():

    fin = open(args.filename, 'r')
    if args.src:
        prefix = 'src.%s.%s' % (args.setid, args.src_lang)
    else:
        prefix = 'ref.%s.%s' % (args.setid, args.trg_lang)

    fout = open(prefix + '.sgm', 'w')
    count = 1

    # write headers
    if args.src:
        fout.write('<srcset setid="%s" srclang="any">\n' % args.setid)
        fout.write('<doc docid="4444" genre="news" origlang="zh">\n')
    else:
        fout.write('<refset trglang="%s" setid="%s" srclang="any">\n'
                   % (args.trg_lang, args.setid))
        fout.write('<doc sysid="ref" docid="4444" genre="news" origlang="zh">')

    fout.write('<p>\n')

    # read line
    while 1:
        line = fin.readline()
        if not line:
            break
        newline = '<seg id="%d"> %s </seg>\n' % (count, line.replace('\n', ''))
        fout.write(newline)
        count += 1

    fout.write('</p>\n')

    # closing tags for headers
    fout.write('</doc>\n')
    if args.src:
        fout.write('</srcset>\n')
    else:
        fout.write('</refset>\n')

    fin.close()
    fout.close()

if __name__ == '__main__':
    main_loop()
