from __future__ import print_function
from inspect import currentframe, getframeinfo

class Log(object):
    def DEBUG(message):
        print(message)
        
    def WARNING(message):
        cf = currentframe()
        line_number = cf.f_back.f_lineno
        file_name = cf.f_back.filename
        if file_name!=None and line_number!=None:
            print_colored(message+", at "+ file_name+ " line-number:"+ line_number,bg=color.bg.yellow)
        elif file_name!=None:
            print_colored(message+", at "+ file_name,bg=color.bg.yellow)
        else:
            print_colored(message,bg=color.bg.yellow)
    def ERROR(message):
        cf = currentframe()
        line_number = cf.f_back.f_lineno
        file_name = cf.f_back.filename
        if file_name!=None and line_number!=None:
            print_colored(message+", at "+ file_name+ " line-number:"+ line_number,bg=color.bg.red)
        elif file_name!=None:
            print_colored(message+", at "+ file_name,bg=color.bg.red)
        else:
            print_colored(message,bg=color.bg.red)
    def LOG(message,file=None):
        if type(out_file) == file:
            out_file.write(message)
            out_file.write("\n")
        elif type(out_file) == str:
            with open(out_file,"a+") as ofile:
                ofile.write(message)
                ofile.write("\n")
        else:
            frameinfo = getframeinfo(currentframe())
            Log.WARNING("Out file should be either str or file object")
    def print_colored(message,fg=color.fg.black,bg=color.bg.white,style="6"):
        format = ';'.join([str(style), str(fg), str(bg)])
        s1 = '\x1b[%sm %s \x1b[0m' % (format, message)
        print(s1)



class colors:
    '''Colors class:
    reset all colors with colors.reset
    two subclasses fg for foreground and bg for background.
    use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.green
    also, the generic bold, disable, underline, reverse, strikethrough,
    and invisible work with the main class
    i.e. colors.bold
    source https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
    '''
    reset='00'
    bold='01'
    disable='02'
    underline='04'
    reverse='07'
    strikethrough='09'
    invisible='08'
    class fg:
        black='30'
        red='31'
        green='32'
        orange='33'
        blue='34'
        purple='35'
        cyan='36'
        lightgrey='37'
        darkgrey='90'
        lightred='91'
        lightgreen='92'
        yellow='93'
        lightblue='94'
        pink='95'
        lightcyan='96',
        white = 50
    class bg:
        black='40'
        red='41'
        green='42'
        orange='43'
        blue='44'
        purple='45'
        cyan='46'
        lightgrey='47',
        white = 50