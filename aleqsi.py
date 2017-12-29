"""Python interface to Aqsis <http://www.aqsis.org/> command-line tools.
Rather than directly wrap the libaqsisxxx libraries, this spawns subprocesses
running the tools.

Also needed: “convert” command from ImageMagick/GraphicsMagick.
"""

import sys
import enum
from numbers import \
    Real
import os
import array
import re
import subprocess
import tempfile
import shutil
import shlex
from qahirah import \
    Colour

@enum.unique
class SEARCH_TYPE(enum.Enum) :
    "search paths for finding various external files."
    SOURCE = 0
    ARCHIVE = 1
    SHADER = 2
    PROCEDURAL = 3
    RESOURCE = 4
    TEXTURE = 5
#end SEARCH_TYPE

class RManSyntaxError(SyntaxError) :

    def __init__(self, *args) :
        super().__init__(*args)
    #end __init__

#end RManSyntaxError

@enum.unique
class FILE_TYPE(enum.Enum) :
    RIB = 1
    SHADER = 2
#end FILE_TYPE

@enum.unique
class DISPLAY(enum.Enum) :
    "how to collect rendered output image files, as specified in “Display” directives," \
    " into Context.images."
    NOTHING = 0 # collect none of them
    FRAMEBUFFER = 1 # collect only ones which specify “framebuffer” option
    ALL = 2 # collect all files mentioned in “Display” directives
    AUTO = 3 # automatically generate a “Display” directive which will be collected
#end DISPLAY

class Converter :

    def __init__(self, name, expect_type, conv, not_type = None) :
        self.name = name
        self.expect_type = expect_type
        self.not_type = not_type
        if isinstance(conv, str) :
            fmt = conv
            conv = lambda ctx, x : fmt % x
        else :
            assert callable(conv)
        #end if
        self.conv = conv
    #end __init__

    def check(self, val, descr) :
        if not isinstance(val, self.expect_type) or self.not_type != None and isinstance(val, self.not_type) :
            raise TypeError("%s must be of %s type" % (descr, self.name))
        #end if
    #end check

#end Converter

class ArrayConverter(Converter) :

    def __init__(self, name, elt_conv, fixed_len = None) :
        self.name = name
        assert isinstance(elt_conv, Converter)
        self.elt_conv = elt_conv
        self.fixed_len = fixed_len
    #end __init__

    def check(self, val, descr) :
        if not isinstance(val, (list, tuple)) :
            raise TypeError("%s must be list or tuple %s type" % (descr, self.name))
        #end if
        if self.fixed_len != None and len(val) != self.fixed_len :
            raise TypeError("%s must have %d elements, not %d" % (descr, self.fixed_len, len(val)))
        #end if
        for i, v in enumerate(val) :
            self.elt_conv.check(v, "%s[%d]" % (self.name, i))
        #end if
    #end check

    def conv(self, ctx, val) :
        return \
            "[" + " ".join(self.elt_conv.conv(ctx, v) for v in val) + "]"
    #end conv

#end ArrayConverter

def quote_rman_str(s) :
    result = ["\""]
    for c in s :
        if c in ("\"", "\\") :
            result.append("\\")
        #end if
        result.append(c)
    #end for
    result.append("\"")
    return \
        "".join(result)
#end quote_rman_str

conv_bool = Converter("boolean", bool, "%d")
conv_int = Converter("integer", int, "%d", not_type = bool)
conv_num = Converter("number", Real, lambda ctx, x : "%%.%dg" % ctx.precision % x)
conv_str = Converter("string", str, lambda ctx, s : quote_rman_str(s))
conv_int_array = ArrayConverter("int_array", conv_int)
conv_num_array = ArrayConverter("num_array", conv_num)
conv_str_array = ArrayConverter("int_array", conv_str)

conv_point = ArrayConverter("point", conv_num, 3)

def valid_bytes(b) :
    "does b have a suitable type for an image bytes object."
    return \
        (
            isinstance(b, bytes)
        or
            isinstance(b, bytearray)
        or
            isinstance(b, array.array) and b.typecode == "B"
        )
#end valid_bytes

class Context :
    "context for feeding model/scene and shader definitions to Aqsis and retrieving" \
    " rendered results. You need to instantiate at least one of these."

    __slots__ = \
        (
            "_tempdir",
            "_workdir",
            "_ribfile_nr",
            "_texfile_nr",
            "_imgfile_nr",
            "search_path",
            "shaders",
            "textures",
            "timeout",
            "images",
            "precision",
            "debug",
            "keep_temps",
        )

    def __init__(self) :
        self._tempdir = None
        self.search_path = dict((k, None) for k in SEARCH_TYPE)
        self.shaders = {}
        self.textures = {}
        self.timeout = None
        self._ribfile_nr = 0
        self._texfile_nr = 0
        self._imgfile_nr = 0
        self.images = []
        self.precision = 7
        self.debug = False
        self.keep_temps = False
    #end __init__

    def _collect_display(self, filename) :
        self.images.append \
          (
            subprocess.check_output
              (
                args = ("convert", filename, "png:/dev/stdout"),
                universal_newlines = False,
                timeout = self.timeout
              )
          )
    #end _collect_display

    @enum.unique
    class _GENFILETYPE(enum.Enum) :
        # generated files with names assigned by user.
        SHADER = 1
        TEXTURE = 2

        @property
        def descrname(self) :
            return \
                {
                    type(self).SHADER : "shader",
                    type(self).TEXTURE : "texture",
                }[self]
        #end descrname

        @property
        def extension(self) :
            return \
                {
                    type(self).SHADER : ".sl",
                    type(self).TEXTURE : ".tex",
                }[self]
        #end extension

        @property
        def require_extension(self) :
            return \
                self == type(self).TEXTURE
        #end require_extension

        @property
        def attrname(self) :
            # name of Context instance variable that collects generated files of this type.
            return \
                {
                    type(self).SHADER : "shaders",
                    type(self).TEXTURE : "textures",
                }[self]
        #end attrname

    #end _GENFILETYPE

    class Rib :
        "context for writing a sequence of RIB statements. Do not instantiate" \
        " directly; get from Context.new_rib()."

        __slots__ = \
            (
                "_parent",
                "nr_colour_samples",
                "_last_object_nr",
                "_display",
                "_imgfile_names",
                "_filename",
                "_out",
                "_infilename",
                "_linenr",
                "_display_pat",
                "_readarchive_pat",
            )

        class ObjectHandle :
            __slots__ = ("parent", "index")

            def __init__(self, parent, index) :
                self.parent = parent
                self.index = index
            #end __init__

        #end ObjectHandle

        def __init__(self, parent, filename, display) :
            self._display_pat = re.compile(r"^\s*display\s*(.+)$", flags = re.IGNORECASE)
            self._readarchive_pat = re.compile(r"^\s*readarchive\s*(.+)$", flags = re.IGNORECASE)
            self._parent = parent
            self.nr_colour_samples = 3
            self._last_object_nr = 0
            self._display = display
            self._imgfile_names = []
            self._filename = filename
            self._out = open(filename, "w")
            self._infilename = "<API call>"
            self._linenr = None
            if display == DISPLAY.AUTO :
                imgfile_name = self._new_imgfile_name()
                self._out.write \
                    (
                        "Display \"%(outfile)s\" \"file\" \"rgba\"\n"
                    %
                        {"outfile" : imgfile_name}
                    )
            #end if
        #end __init__

        def _new_imgfile_name(self) :
            # pity Aqsis cannot generate PNG directly...
            self._parent._imgfile_nr += 1
            imgfile_name = os.path.join(self._parent._tempdir, "out%03d.tif" % self._parent._imgfile_nr)
            self._imgfile_names.append(imgfile_name)
            return \
                imgfile_name
        #end _new_imgfile_name

        def writeln(self, line) :

            def do_auto_display() :
                # collects output filenames for automatic display.
                nonlocal line
                display_match = self._display_pat.match(line)
                if display_match != None :
                    display_parms = shlex.split(display_match.group(1))
                else :
                    display_parms = None
                #end if
                if (
                        display_parms != None
                    and
                        len(display_parms) >= 3
                ) :
                    display_mode = display_parms[1].lower()
                    if display_mode in ("file", "tiff", "framebuffer") :
                        imgfile_name = display_parms[0]
                        seen_imgfile = imgfile_name.startswith("+") # assume I’ve seen it before
                        if seen_imgfile :
                            imgfile_name = imgfile_name[1:]
                        #end if
                        if (
                                self._display == DISPLAY.ALL and not seen_imgfile
                            or
                                self._display == DISPLAY.FRAMEBUFFER and display_mode == "framebuffer"
                        ) :
                            self._imgfile_names.append(os.path.join(self._parent._workdir, imgfile_name))
                        #end if
                        if display_mode == "framebuffer" :
                            line = None
                        #end if
                    #end if
                #end if
            #end do_auto_display

            def do_auto_include() :
                nonlocal line
                readarchive_match = self._readarchive_pat.match(line)
                if readarchive_match != None :
                    parms = shlex.split(readarchive_match.group(1))
                    if len(parms) != 1 :
                        raise RManSyntaxError \
                          (
                            "expecting exactly one filename for ReadArchive directive",
                            (self._infilename, self._linenr, None, line)
                          )
                    #end if
                    line = None
                    filename = self._parent.find_file(parms[0], SEARCH_TYPE.ARCHIVE)
                    save_infilename = self._infilename
                    save_linenr = self._linenr
                    self._infilename = filename
                    for line in open(filename, "r") :
                        self._linenr += 1
                        self.writeln(line.rstrip("\n"))
                    #end for
                    self._infilename = save_infilename
                    self._linenr = save_linenr
                #end if
            #end do_auto_include

        #begin writeln
            if self._out == None :
                raise RuntimeError("output RIB stream has been closed")
            #end if
            if self._display != DISPLAY.NOTHING :
                do_auto_display()
            #end if
            if line != None :
                do_auto_include()
            #end if
            if line != None :
                self._out.write(line)
                self._out.write("\n")
            #end if
            return \
                self
        #end writeln

        def _write_stmt(self, stmtname, arglist, kwargs) :
            kwarglist = []
            for k in sorted(kwargs.keys()) : # might as well insert in some predictable order
                kwarglist.extend([quote_rman_str(k), self._parent.convert_general(kwargs[k])])
                  # not enforcing specific expected types for keyword args for now
            #end for
            self._out.write("%s %s\n" % (stmtname, " ".join(arglist + kwarglist)))
            return \
                self
        #end _write_stmt

        def close(self) :
            self._out.close()
            self._out = None
            extra = []
            for search_type in SEARCH_TYPE :
                if search_type != SEARCH_TYPE.SOURCE :
                    path = self._parent.search_path[search_type]
                    if path != None :
                        if search_type == SEARCH_TYPE.RESOURCE :
                            # strange there is no specific command-line option for this
                            opt = "-option=Option \"searchpath\" \"resource \" [\"%s\"]" % path
                        else :
                            opt = \
                              (
                                    "-%s-search=%s"
                                %
                                    (
                                        {
                                            SEARCH_TYPE.ARCHIVE : "archive",
                                            SEARCH_TYPE.SHADER : "shader",
                                            SEARCH_TYPE.PROCEDURAL : "procedural",
                                            SEARCH_TYPE.TEXTURE : "texture",
                                        }[search_type],
                                        path,
                                    )
                              )
                        #end if
                        extra.append(opt)
                    #end if
                #end if
            #end for
            # no need for SEARCH_TYPE.SOURCE here, since I automatically
            # expanded all RIB includes myself
            aqsis_output = subprocess.check_output \
              (
                args = ["aqsis"] + extra + [self._filename],
                stdin = subprocess.DEVNULL,
                stderr = subprocess.STDOUT,
                universal_newlines = True,
                cwd = self._parent._workdir,
                timeout = self._parent.timeout
              )
            if self._parent.debug :
                sys.stderr.write(aqsis_output)
            #end if
            if len(self._imgfile_names) != 0 :
                for imgfile_name in self._imgfile_names :
                    self._parent._collect_display(imgfile_name)
                #end for
            #end if
        #end close

        def collect_display(self, filename) :
            "tells me that a “display” directive has written the file filename," \
            " so its contents should be collected as another of the images to be" \
            " returned by the parent Context."
            self._parent._collect_display(os.path.join(self._parent._workdir, filename))
            return \
                self
        #end collect_display

        # all the rest of the RenderMan statement-generating methods
        # are defined by calling def_rman_stmt (below), except the following

        def colour_samples(self, to_rgb, from_rgb) :
            if (
                    not isinstance(to_rgb, (list, tuple))
                or
                    not isinstance(from_rgb, (list, tuple))
                or
                    len(to_rgb) != len(from_rgb)
                or
                    len(to_rgb) % 3 != 0
                or
                    len(to_rgb) == 0
            ) :
                raise TypeError("args must be arrays of equal nonzero size, being a multiple of 3")
            #end if
            self.nr_colour_samples = len(to_rgb) // 3
            self._write_stmt("ColorSamples", [conv_num_array.conv(self._parent, to_rgb), conv_num_array.conv(self._parent, from_rgb)], {})
        #end colour_samples

        def colour(self, *args) :
            opacity = None
            if self.nr_colour_samples == 3 and len(args) == 1 and isinstance(args[0], Colour) :
                colour = args[0]
                opacity = colour.a
                args = colour[:3]
            elif len(args) != self.nr_colour_samples or not all(isinstance(c, Real) for c in args) :
                raise TypeError("expecting %d float args" % self.nr_colour_samples)
            #end if
            self._write_stmt("Color", [conv_num.conv(self._parent, c) for c in args], {})
            if opacity != None :
                self.opacity(opacity)
            #end if
        #end colour

        def opacity(self, *args) :
            if len(args) != self.nr_colour_samples and len(args) != 1 or not all(isinstance(c, Real) for c in args) :
                raise TypeError("expecting 1 or %d float args" % self.nr_colour_samples)
            #end if
            if len(args) == 1 :
                args = [args[0]] * self.nr_colour_samples
            #end if
            self._write_stmt("Opacity", [conv_num.conv(self._parent, c) for c in args], {})
        #end opacity

        def basis(self, ubasis, ustep, vbasis, vstep) :
            if not all \
              (
                isinstance(b, str) or isinstance(b, (tuple, list)) and len(b) == 16
                for b in (ubasis, vbasis)
              ) :
                raise TypeError("ubasis and vbasis must be basis names or 16-element matrices")
            #end if
            bases = tuple \
              (
                (conv_num_array, conv_str)[isinstance(b, str)].conv(self._parent, b)
                for b in (ubasis, vbasis)
              )
            self._write_stmt("Basis", [bases[0], conv_num.conv(self._parent, ustep), bases[1], conv_num.conv(self._parent, vstep)], {})
        #end basis

        def object_begin(self) :
            "returns ObjectHandle, not self!"
            self._last_object_nr += 1
            object_nr = self._last_object_nr
            self._out.write("ObjectBegin %d\n" % object_nr)
            return \
                self.ObjectHandle(self, object_nr)
        #end object_begin

        def object_instance(self, handle) :
            if not isinstance(handle, self.ObjectHandle or handle.parent != self) :
                raise TypeError("handle should be one of my ObjectInstance objects")
            #end if
            self._out.write("ObjectInstance %d\n" % handle.index)
            return \
                self
        #end object_instance

    #end Rib

    class Shader :
        "context for writing a shader definition. Do not instantiate" \
        " directly; get from Context.new_shader()."

        __slots__ = ("_parent", "_filename", "_out")

        def __init__(self, parent, filename, out) :
            self._parent = parent
            self._filename = filename
            self._out = out
        #end __init__

        def write(self, s) :
            "writes more shader source."
            if self._out == None :
                raise RuntimeError("output shader stream has been closed")
            #end if
            self._out.write(s)
            return \
                self
        #end write

        def writeln(self, stmt) :
            "writes a line of shader source."
            if self._out == None :
                raise RuntimeError("output shader stream has been closed")
            #end if
            self._out.write(stmt)
            self._out.write("\n")
            return \
                self
        #end writeln

        def close(self) :
            "terminates writing of shader source and compiles the shader."
            self._out.close()
            self._out = None
            self._parent._compile_shader(self._filename)
        #end close

    #end Shader

    def _init_temp(self) :
        # ensures the temporary directory structure has been created.
        if self._tempdir == None :
            self._tempdir = tempfile.mkdtemp(prefix = "aleqsi-")
            self._workdir = os.path.join(self._tempdir, "work")
            os.mkdir(self._workdir)
              # separate subdirectory for files created by caller
        #end if
    #end _init_temp

    def _conv_arg(self, stmtname, index, val, conv) :
        conv.check(val, "arg %d to %s" % (index, stmtname))
        return \
            conv.conv(self, val)
    #end _conv_arg

    def convert_general(self, val) :
        if isinstance(val, (list, tuple)) :
            convert = lambda ctx, val : "[" + " ".join(ctx.convert_general(x) for x in val) + "]"
        elif isinstance(val, bool) :
            convert = conv_bool.conv
        elif isinstance(val, (int, float)) :
            convert = conv_num.conv
        elif isinstance(val, str) :
            convert = conv_str.conv
        else :
            raise TypeError("cannot convert arg type %s" % repr(type(val)))
        #end if
        return \
            convert(self, val)
    #end convert_general

    def cleanup(self) :
        if self._tempdir != None :
            if not self.keep_temps :
                try :
                    shutil.rmtree(self._tempdir)
                except OSError :
                    pass
                #end try
            #end if
            self._tempdir = None
        #end if
    #end cleanup

    def __del__(self) :
        self.cleanup()
    #end __del__

    def _new_genfile(self, filetype, name) :
        items = getattr(self, filetype.attrname)
        if name in items :
            raise KeyError("duplicate %s name “%s”" % (filetype.descrname, name))
        #end if
        if filetype.require_extension :
            if not name.endswith(filetype.extension) :
                raise RManSyntaxError \
                  (
                    "%s name must end with %s" % (filetype.descrname, filetype.extension)
                  )
            #end if
            basename = name[:- len(filetype.extension)]
        else :
            basename = name
        #end if
        if "/" in basename or "." in basename :
            raise RManSyntaxError("no dots or slashes allowed in name")
        #end if
        self._init_temp()
        filename = os.path.join(self._workdir, basename + filetype.extension)
        items[name] = filename
        return \
            filename
    #end _new_genfile

    def new_rib(self, display = DISPLAY.FRAMEBUFFER) :
        if not isinstance(display, DISPLAY) :
            raise TypeError("display option must be one of the DISPLAY.xxx values")
        #end if
        self._init_temp()
        self._ribfile_nr += 1
        outfile_name = os.path.join(self._tempdir, "in%03d.rib" % self._ribfile_nr)
        return \
            self.Rib(self, outfile_name, display)
    #end new_rib

    def _compile_rib(self, iter, display, infilename) :
        rib = self.new_rib(display)
        # need to copy line by line to allow autoparsing of display and include lines
        rib._infilename = infilename
        rib._linenr = 0
        for line in iter :
            rib._linenr += 1
            rib.writeln(line.rstrip("\n"))
        #end for
        rib.close()
    #end _compile_rib

    def compile_rib(self, iter, display = DISPLAY.FRAMEBUFFER) :
        self._compile_rib(iter, display, "<iterator>")
    #end compile_rib

    def compile_rib_file(self, filename, display = DISPLAY.FRAMEBUFFER) :
        fullpath = self.find_file(filename, SEARCH_TYPE.SOURCE)
        self._compile_rib(open(fullpath, "r"), display, filename)
    #end compile_rib_file

    def _compile_shader(self, filename) :
        slproc = subprocess.Popen \
          (
            args = ("aqsl", filename),
            stdin = subprocess.DEVNULL,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            universal_newlines = True,
            cwd = self._workdir
          )
        slproc_output, _ = slproc.communicate(timeout = self.timeout)
        if slproc.returncode == 0 :
            if self.debug and slproc_output != None :
                sys.stderr.write(slproc_output)
            #end if
        else :
            raise RManSyntaxError \
              (
                "shader compilation failed with code %d" % slproc.returncode,
                (filename, 0, None, slproc_output)
              )
        #end if
    #end _compile_shader

    def new_shader(self, name) :
        shader_filename = self._new_genfile(self._GENFILETYPE.SHADER, name)
        return \
            self.Shader(self, shader_filename, open(shader_filename, "w"))
    #end new_shader

    def compile_shader_file(self, filename) :
        name = os.path.basename(filename)
        name, ext = os.path.splitext(name)
        if ext != ".sl" :
            raise RManSyntaxError("shader filename does not end in .sl")
        #end if
        fullpath = self.find_file(filename, SEARCH_TYPE.SOURCE)
        shader = self.new_shader(name)
        # instead of copying, should I compile direct from original source file?
        for line in open(fullpath, "r") :
            shader.write(line)
        #end for
        shader.close()
    #end compile_shader_file

    def _teqser_args(self, kwargs, doing_envcube) :
        valid_opts = \
            {
                "bake" : True,
                "compression" : True,
                # "envcube" handled specially
                "envlatl" : False,
                "filter" : True,
                "envcube_fov" : True,
                "quality" : True,
                "shadow" : False,
                "swrap" : True,
                "twrap" : True,
                "wrap" : True,
                "swidth" : True,
                "twidth" : True,
                "width" : True,
                # todo: verbose
            }
        cmd = ["teqser"]
        unprocessed = set(kwargs.keys())
        for opt in sorted(valid_opts.keys()) :
            if opt in kwargs :
                has_value = valid_opts[opt]
                if has_value :
                    value = kwargs[opt]
                    if value != None :
                        value = "=%s" % value
                    #end if
                else :
                    value = kwargs[opt]
                    if not isinstance(value, bool) :
                        raise TypeError("value of %s arg must be boolean" % opt)
                    #end if
                    value = (None, "")[value]
                #end if
                if value != None :
                    cmd.append("-%s%s" % (opt, value))
                #end if
                unprocessed.remove(opt)
            #end if
        #end for
        if len(unprocessed) != 0 :
            raise TypeError("unrecognized keyword args: %s" % ", ".join(sorted(unprocessed)))
        #end if
        if doing_envcube :
            cmd.append("-envcube")
        #end if
        return \
            cmd
    #end _teqser_args

    def define_texture(self, name, *args, **kwargs) :
        output_file = self._new_genfile(self._GENFILETYPE.TEXTURE, name)
        if len(args) == 1 and isinstance(args[0], (list, tuple)) :
            args = args[0]
        #end if
        doing_envcube = None
        if len(args) == 6 :
            doing_envcube = True
        elif len(args) == 1 :
            doing_envcube = False
        #end if
        if doing_envcube == None or not all(valid_bytes(b) for b in args) :
            raise TypeError("positional args must be 1 or 6 bytes objects")
        #end if
        input_files = []
        for b in args :
            filename = self._new_texfile_name()
            if True :
                pngtemp = filename + ".png"
                pngout = open(pngtemp, "wb")
                pngout.write(b)
                pngout.flush()
                subprocess.check_call \
                  (
                    args = ("convert", pngtemp, filename),
                    universal_newlines = False,
                    timeout = self.timeout
                  )
            else :
                # feeding PNG byte stream directly via pipe doesn’t seem to work
                # -- convert complains with “insufficient image data”
                subprocess.check_call \
                  (
                    args = ("convert", "png:/dev/stdin", filename),
                    input = b,
                    universal_newlines = False,
                    timeout = self.timeout
                  )
            #end if
            input_files.append(filename)
        #end for
        teqser_output = subprocess.check_output \
          (
            args = self._teqser_args(kwargs, doing_envcube) + input_files + [output_file],
            stdin = subprocess.DEVNULL,
            stderr = subprocess.STDOUT,
            universal_newlines = True,
            cwd = self._workdir,
            timeout = self.timeout
          )
        if self.debug :
            sys.stderr.write(teqser_output)
        #end if
    #end define_texture

    def define_texture_file(self, name, *args, **kwargs) :
        output_file = self._new_genfile(self._GENFILETYPE.TEXTURE, name)
        if len(args) == 1 and isinstance(args[0], (list, tuple)) :
            args = args[0]
        #end if
        if len(args) == 6 :
            doing_envcube = True
        elif len(args) == 1 :
            doing_envcube = False
        else :
            raise TypeError("positional args must be 1 or 6 file names")
        #end if
        input_files = list(self.find_file(f, SEARCH_TYPE.TEXTURE) for f in args)
        teqser_output = subprocess.check_output \
          (
            args = self._teqser_args(kwargs, doing_envcube) + input_files + [output_file],
            stdin = subprocess.DEVNULL,
            stderr = subprocess.STDOUT,
            universal_newlines = True,
            cwd = self._workdir,
            timeout = self.timeout
          )
        if self.debug :
            sys.stderr.write(teqser_output)
        #end if
    #end define_texture_file

    def set_search_path(self, search_type, items) :
        if not isinstance(search_type, SEARCH_TYPE) :
            raise TypeError("search_type must be one of the SEARCH_TYPE enum values.")
        #end if
        old_value = self.search_path[search_type]
        if old_value == None :
            old_value = "&"
        #end if
        collect = []
        for item in items.split(":") :
            if item == "&" :
                collect.extend(old_value.split(":"))
            else :
                collect.append(item)
            #end if
        #end for
        new_value = ":".join(collect)
        self.search_path[search_type] = new_value
    #end set_search_path

    def find_file(self, filename, search_type, must_exist = True) :
        if not isinstance(search_type, SEARCH_TYPE) :
            raise TypeError("search_type must be one of the SEARCH_TYPE enum values.")
        #end if
        file_arg = filename
        if not file_arg.startswith("/") :
            search1 = self.search_path[search_type]
            if search_type != SEARCH_TYPE.RESOURCE :
                search2 = self.search_path[SEARCH_TYPE.RESOURCE]
            else :
                search2 = None
            #end if
            try_path = []
            for search in (search1, search2) :
                if search != None :
                    for try_dir in search.split(":") :
                        if try_dir == "&" :
                            self._init_temp()
                            try_path.append(os.path.join(self._workdir, file_arg))
                        else :
                            try_path.append(os.path.join(try_dir, file_arg))
                        #end if
                    #end for
                #end if
            #end for
            if len(try_path) == 0 :
                try_path = [os.path.join(self._workdir, file_arg)]
            #end if
        else :
            try_path = [os.path.join(self._workdir, file_arg)]
        #end if
        while True :
            if len(try_path) == 0 :
                if must_exist :
                    raise RuntimeError("cannot find file “%s”" % filename)
                #end if
                file_arg = None
                break
            #end if
            file_arg = try_path.pop(0)
            if os.path.exists(file_arg) :
                break
            #end if
        #end while
        return \
            file_arg
    #end find_file

    def _new_texfile_name(self) :
        # pity Aqsis cannot accept PNG directly...
        self._texfile_nr += 1
        texfile_name = os.path.join(self._tempdir, "tex%03d.tif" % self._texfile_nr)
        return \
            texfile_name
    #end _new_texfile_name

#end Context

def def_rman_stmt(methname, stmtname, argtypes) :

    def gen_stmt(self, *args, **kwargs) :
        if len(args) == len(argtypes) + 1 and len(kwargs) == 0 and isinstance(args[-1], dict) :
            kwargs = args[-1]
            args = args[:-1]
        #end if
        if len(args) == 1 and isinstance(args[0], (list, tuple)) and len(argtypes) != 1 :
            args = args[0]
        #end if
        if len(args) != len(argtypes) :
            raise TypeError("stmt %s expects %d positional args" % (stmtname, len(argtypes)))
        #end if
        arglist = list \
          (
            self._parent._conv_arg(stmtname, i + 1, val, conv)
            for i, (val, conv) in enumerate(zip(args, argtypes))
          )
        return \
            self._write_stmt(stmtname, arglist, kwargs)
    #end gen_stmt

#begin def_rman_stmt
    gen_stmt.__name__ = methname
    gen_stmt.__doc__ = "generates a RenderMan “%s” statement." % stmtname
    setattr(Context.Rib, methname, gen_stmt)
#end def_rman_stmt

vector_arg = [conv_num] * 3
matrix_arg = [conv_num] * 16
for methname, stmtname, argtypes in \
    (
        ("declare", "Declare", [conv_str, conv_str]),

        ("begin", "Begin", [conv_str]),
        ("end", "End", []),
        ("context", "Context", [conv_int]),
        ("frame_begin", "FrameBegin", [conv_int]),
        ("frame_end", "FrameEnd", []),
        ("world_begin", "WorldBegin", []),
        ("world_end", "WorldEnd", []),

        ("format", "Format", [conv_int, conv_int, conv_num]),
        ("frame_aspect_ratio", "FrameAspectRatio", [conv_num]),
        ("screen_window", "ScreenWindow", [conv_num, conv_num, conv_num, conv_num]),
        ("crop_window", "CropWindow", [conv_num, conv_num, conv_num, conv_num]),
        ("projection", "Projection", [conv_str]),
        ("clipping", "Clipping", [conv_num, conv_num]),
        ("clipping_plane", "ClippingPlane", [conv_num, conv_num, conv_num, conv_num, conv_num, conv_num]),
        ("depth_of_field", "DepthOfField", [conv_num, conv_num, conv_num]),
        ("shutter", "Shutter", [conv_num, conv_num]),
        ("pixel_variance", "PixelVariance", [conv_num]),
        ("pixel_samples", "PixelSamples", [conv_num, conv_num]),
        ("pixel_filter", "PixelFilter", [conv_str, conv_num, conv_num]),
        ("exposure", "Exposure", [conv_num, conv_num]),
        ("imager", "Imager", [conv_str]),
        ("quantize", "Quantize", [conv_str, conv_int, conv_int, conv_int, conv_num]),
        # ("display", "Display") TBD
        ("hider", "Hider", [conv_str]),
        # ("colour_samples", "ColorSamples") treated specially

        ("option", "Option", [conv_str]),
        ("attribute_begin", "AttributeBegin", []),
        ("attribute_end", "AttributeEnd", []),
        # ("colour", "Color") treated specially
        # ("opacity", "Opacity") treated specially

        ("texture_coordinates", "TextureCoordinates", [conv_num, conv_num, conv_num, conv_num, conv_num, conv_num, conv_num, conv_num]),
        ("light_source", "LightSource", [conv_str, conv_int]),
        ("area_light_source", "AreaLightSource", [conv_str, conv_int]),
        ("illuminate", "Illuminate", [conv_int, conv_bool]),
        ("surface", "Surface", [conv_str]),
        ("displacement", "Displacement", [conv_str]),
        ("atmosphere", "Atmosphere", [conv_str]),
        ("interior", "Interior", [conv_str]),
        ("exterior", "Exterior", [conv_str]),

        ("shading_rate", "ShadingRate", [conv_num]),
        ("shading_interpolation", "ShadingInterpolation", [conv_str]),
        ("matte", "Matte", [conv_bool]),
        ("bound", "Bound", [conv_num, conv_num, conv_num, conv_num, conv_num, conv_num]),
        ("detail", "Detail", [conv_num, conv_num, conv_num, conv_num, conv_num, conv_num]),
        ("detail_range", "DetailRange", [conv_num, conv_num, conv_num, conv_num]),
        ("geometric_approximation", "GeometricApproximation", [conv_str, conv_num]),
        ("orientation", "Orientation", [conv_str]),
        ("reverse_orientation", "ReverseOrientation", []),
        ("sides", "Sides", [conv_int]),

        ("identity", "Identity", []),
        ("transform", "Transform", matrix_arg),
        ("concat_transform", "ConcatTransform", matrix_arg),
        ("translate", "Translate", vector_arg),
        ("rotate", "Rotate", [conv_num] + vector_arg),
        ("scale", "Scale", vector_arg),
        ("skew", "Skew", [conv_num, conv_num, conv_num, conv_num, conv_num, conv_num, conv_num]),
        ("coordinate_system", "CoordinateSystem", [conv_str]),
        ("coord_sys_transform", "CoordSysTransform", [conv_str]),
        # TransformPoints probably not useful
        ("transform_begin", "TransformBegin", []),
        ("transform_end", "TransformEnd", []),

        ("attribute", "Attribute", [conv_str]),
        ("attribute_begin", "AttributeBegin", []),
        ("attribute_end", "AttributeEnd", []),

        ("polygon", "Polygon", [conv_int]),
        ("general_polygon", "GeneralPolygon", [conv_int_array]),
        ("points_polygons", "PointsPolygons", [conv_int_array, conv_int_array]),
        ("points_general_polygons", "PointsGeneralPolygons", [conv_int_array, conv_int_array, conv_int_array]),

        # ("basis", "Basis") handled specially
        ("patch", "Patch", [conv_str]),
        ("patch_mesh", "PatchMesh", [conv_str, conv_int, conv_str, conv_int, conv_int]),
        ("nu_patch", "NuPatch", [conv_int, conv_int, conv_num_array, conv_num, conv_num, conv_int, conv_int, conv_num_array, conv_num, conv_num]),
        ("trim_curve", "TrimCurve", [conv_int, conv_int_array, conv_int_array, conv_num_array, conv_num, conv_num, conv_int_array, conv_num_array, conv_num_array, conv_num_array]),

        ("subdivision_mesh", "SubdivisionMesh", [conv_str, conv_int, conv_int_array, conv_int_array, conv_int, conv_str_array, conv_int_array, conv_int_array, conv_num_array]),

        ("sphere", "Sphere", [conv_num, conv_num, conv_num, conv_num]),
        ("cone", "Cone", [conv_num, conv_num, conv_num]),
        ("cylinder", "Cylinder", [conv_num, conv_num, conv_num, conv_num]),
        ("hyperboloid", "Hyperboloid", [conv_point, conv_point, conv_num]),
        ("paraboloid", "Paraboloid", [conv_num, conv_num, conv_num, conv_num]),
        ("disk", "Disk", [conv_num, conv_num, conv_num]),
        ("torus", "Torus", [conv_num, conv_num, conv_num, conv_num, conv_num]),

        ("points", "Points", [conv_int]),
        ("curves", "Curves", [conv_str, conv_int_array, conv_str]),
        ("blobby", "Blobby", [conv_int, conv_int, conv_int_array, conv_num, conv_num_array, conv_int, conv_str_array]),

        # ("procedural", "Procedural" [conv_str, conv_str_array, bound TBD]),
        ("geometry", "Geometry", [conv_str]),

        ("solid_begin", "SolidBegin", [conv_str]),
        ("solid_end", "SolidEnd", []),
        # ("object_begin", "ObjectBegin") handled specially
        ("object_end", "ObjectEnd", []),
        # ("object_instance", "ObjectInstance") handled specially

        # ("motion_begin", "MotionBegin", [conv_int, conv_num, conv_num ... TBD]),
        ("motion_end", "MotionEnd", []),

        # TBD section 7.1 texture-map utilities?

        ("error_handler", "ErrorHandler", [conv_str]),
        # ("read_archive, "ReadArchive", [conv_str]), TBD read and include lines myself
    ) \
:
    def_rman_stmt(methname, stmtname, argtypes)
#end for
del methname, stmtname, argtypes
del vector_arg, matrix_arg

del def_rman_stmt # your work is done
