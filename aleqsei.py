"""Python interface to Aqsis <http://www.aqsis.org/> command-line tools.
Rather than directly wrap the libaqsisxxx libraries, this spawns subprocesses
running the tools.

Also needed: “convert” command from ImageMagick/GraphicsMagick.
"""
#+
# Copyright 2020 Lawrence D'Oliveiro <ldo@geek-central.gen.nz>.
# Licensed under the GNU Lesser General Public License v2.1 or later.
#-

import sys
import enum
from numbers import \
    Real
import os
import array
import re
import subprocess
import tempfile
import gzip
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
conv_str_array = ArrayConverter("str_array", conv_str)

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

    @staticmethod
    def _open_read(filename) :
        # opens a (possibly compressed) text file for reading.
        return \
            (open, gzip.open)[filename.endswith(".gz")](filename, "rt")
    #end _open_read

    def _collect_display(self, filename) :
        if os.path.isfile(filename) :
            self.images.append \
              (
                subprocess.check_output
                  (
                    args = ("convert", filename, "png:-"),
                    universal_newlines = False,
                    timeout = self.timeout
                  )
              )
        else :
            raise UserWarning("missing image file: %s" % repr(filename))
        #end if
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
                "_CtxBlock",
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

            class _CtxBlock :

                rib = self

                def __init__(self, prefix, begin_args, begin_kwargs) :
                    self.prefix = prefix
                    self.begin_args = begin_args
                    self.begin_kwargs = begin_kwargs
                #end __init__

                def __enter__(self) :
                    self.rib._write_stmt("%sBegin" % self.prefix, self.begin_args, self.begin_kwargs)
                    return \
                        self
                #end __enter__

                def __exit__(self, exception_type, exception_value, traceback) :
                    self.rib._write_stmt("%sEnd" % self.prefix, [], {})
                #end __exit__

            #end _CtxBlock

        #begin __init__
            self._CtxBlock = _CtxBlock
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
                    self.display(*display_parms[0:3])
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
                    self.read_archive(parms[0])
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
                print("close: to convert: %s" % repr(self._imgfile_names)) # debug
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

        def display(self, name, display_type, display_mode) :
            # todo: extra parms?
            if display_type in ("file", "tiff", "framebuffer") :
                seen_imgfile = name.startswith("+") # assume I’ve seen it before
                if seen_imgfile :
                    name = name[1:]
                #end if
                if (
                        self._display == DISPLAY.ALL and not seen_imgfile
                    or
                        self._display == DISPLAY.FRAMEBUFFER and display_type == "framebuffer"
                ) :
                    self._imgfile_names.append(os.path.join(self._parent._workdir, name))
                #end if
                if display_type == "framebuffer" :
                    line = None
                #end if
            #end if
        #end display

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
            return \
                self
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
            return \
                self
        #end colour

        def opacity(self, *args) :
            if len(args) != self.nr_colour_samples and len(args) != 1 or not all(isinstance(c, Real) for c in args) :
                raise TypeError("expecting 1 or %d float args" % self.nr_colour_samples)
            #end if
            if len(args) == 1 :
                args = [args[0]] * self.nr_colour_samples
            #end if
            self._write_stmt("Opacity", [conv_num.conv(self._parent, c) for c in args], {})
            return \
                self
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
            return \
                self
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

        def motion(self, steps, *cmds) :
            "defines moving items for purposes of computing motion blur." \
            " steps is a sequence of 2 or more time values, and each of *cmds" \
            " is a 2-tuple of («cmd», «args_seq») where «cmd» is a command name" \
            " and «args_seq» is a sequence the same length as steps, each" \
            " element of which is a set of arguments to «cmd»."
            # see <https://renderman.pixar.com/resources/RenderMan_20/graphicsState.html#rimotionbegin>
            Rib = type(self)
            valid_cmds = \
                { # ones allowed inside a MotionBegin/MotionEnd block
                    "transform",
                    "concat_transform",
                    "perspective",
                    "translate",
                    "rotate",
                    "scale",
                    "skew",

                    "projection",
                    "displacement",

                    "bound",
                    "detail",

                    "polygon",
                    "general_polygon",
                    "points_polygons",
                    "points_general_polygons",
                    "patch",
                    "patch_mesh",
                    "nu_patch",
                    "sphere",
                    "cone",
                    "cylinder",
                    "hyperboloid",
                    "paraboloid",
                    "disk",
                    "torus",
                    "points",
                    "curves",
                    "subdivision_mesh",
                    "blobby",

                    "colour",
                    "opacity",

                    "light_source",
                    "area_light_source",

                    "surface",
                    "interior",
                    "exterior",
                    "atmosphere",
                }
            if (
                    not isinstance(steps, (list, tuple))
                or
                    len(steps) < 2
                or
                    not all(isinstance(x, Real) for x in steps)
            ) :
                raise TypeError("steps must be a sequence of 2 or more numbers.")
            #end if
            if (
                    not isinstance(cmds, (list, tuple))
                or
                    not all(isinstance(elt, (list, tuple)) and len(elt) == 2 for elt in cmds)
                or
                    not all
                      (
                            isinstance(elt[0], str)
                        and
                            elt[0] in valid_cmds
                        and
                            isinstance(elt[1], (list, tuple))
                        and
                            len(elt[1]) == len(steps)
                        for elt in cmds
                      )
            ) :
                raise TypeError("cmds must be sequence of («cmd», «args_seq»).")
            #end if
            for cmd, args_seq in cmds :
                self._write_stmt("MotionBegin", [conv_num_array.conv(self._parent, steps)], {})
                meth = getattr(self, cmd)
                for args in args_seq :
                    meth(*args)
                #end for
                self._write_stmt("MotionEnd", [], {})
            #end for
        #end motion

        def procedural(self, procname, args, bound) :
            # documented here: <https://renderman.pixar.com/resources/RenderMan_20/geometricPrimitives.html#riprocedural>
            if (
                    not isinstance(args, (list, tuple))
                or
                    not isinstance(bound, (list, tuple))
                or
                    not all(isinstance(a, str) for a in args)
                or
                    len(bound) != 6
                or
                    not all(isinstance(x, Real) for x in bound)
            ) :
                raise TypeError("args must be array of strings and bound must be array of 6 numbers")
            #end if
            self._write_stmt \
              (
                "Procedural",
                [
                    conv_str.conv(self._parent, procname),
                    conv_str_array.conv(self._parent, args),
                    conv_num_array.conv(self._parent, bound),
                ],
                {}
              )
        #end procedural

        def read_archive(self, filename) :
            filename = self._parent.find_file(filename, SEARCH_TYPE.ARCHIVE)
            save_infilename = self._infilename
            save_linenr = self._linenr
            try :
                self._infilename = filename
                self._linenr = 0
                for line in self._parent._open_read(filename) :
                    self._linenr += 1
                    self.writeln(line.rstrip("\n"))
                #end for
            finally :
                self._infilename = save_infilename
                self._linenr = save_linenr
            #end try
        #end read_archive

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

    def _compile_rib(self, src, display, infilename) :
        # need to copy line by line to allow autoparsing of display and include lines
        if isinstance(src, str) :
            src_iter = iter(src.split("\n"))
        elif hasattr(src, "__next__") or hasattr(src, "__iter__") :
            src_iter = src # fine
        else :
            raise TypeError("src must be string or iterable of strings")
        #end if
        rib = self.new_rib(display)
        rib._infilename = infilename
        rib._linenr = 0
        for line in src_iter :
            rib._linenr += 1
            rib.writeln(line.rstrip("\n"))
        #end for
        rib.close()
    #end _compile_rib

    def compile_rib(self, src, display = DISPLAY.FRAMEBUFFER) :
        self._compile_rib(src, display, "<iterator>")
    #end compile_rib

    def compile_rib_file(self, filename, display = DISPLAY.FRAMEBUFFER) :
        fullpath = self.find_file(filename, SEARCH_TYPE.SOURCE)
        self._compile_rib(self._open_read(fullpath), display, filename)
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

    def compile_shader(self, name, src) :
        shader = self.new_shader(name)
        if isinstance(src, str) :
            shader.write(src)
        elif hasattr(src, "__next__") or hasattr(src, "__iter__") :
            for line in src :
                shader.write(line)
            #end for
        else :
            raise TypeError("src must be string or iterable of strings")
        #end if
        shader.close()
    #end compile_shader

    def compile_shader_file(self, filename) :
        name = os.path.basename(filename)
        name, ext = os.path.splitext(name)
        if ext != ".sl" :
            raise RManSyntaxError("shader filename does not end in .sl")
        #end if
        fullpath = self.find_file(filename, SEARCH_TYPE.SOURCE)
        shader = self.new_shader(name)
        # instead of copying, should I compile direct from original source file?
        for line in self._open_read(fullpath) :
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
        self._init_temp()
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

def def_rman_stmt(methname, stmtname, argtypes, is_block) :

    def process_arglist(self, stmtname, argtypes, args, kwargs) :
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
        return \
            (
                list
                  (
                    self._parent._conv_arg(stmtname, i + 1, val, conv)
                    for i, (val, conv) in enumerate(zip(args, argtypes))
                  ),
                kwargs
            )
    #end process_arglist

    def def_simple_stmt(methname, stmtname, argtypes) :

        def gen_stmt(self, *args, **kwargs) :
            arglist, kwargs = process_arglist(self, stmtname, argtypes, args, kwargs)
            return \
                self._write_stmt(stmtname, arglist, kwargs)
        #end gen_stmt

    #begin def_simple_stmt
        gen_stmt.__name__ = methname
        gen_stmt.__doc__ = "generates a RenderMan “%s” statement." % stmtname
        setattr(Context.Rib, methname, gen_stmt)
    #end def_simple_stmt

    def def_context_stmts() :

        prefix = methname + ("", "_")[len(methname) != 0]
        context_methname = prefix + "context"

        def gen_context_create(self, *args, **kwargs) :
            arglist, kwargs = process_arglist(self, context_methname, argtypes, args, kwargs)
            return \
                self._CtxBlock(stmtname, arglist, kwargs)
        #end gen_context_create

    #begin def_context_stmts
        gen_context_create.__name__ = prefix + "context"
        gen_context_create.__doc__ = \
            (
                "returns a context manager that generates paired RenderMan “%(prefix)sBegin”"
                " and “%(prefix)sEnd” statements."
            %
                {"prefix" : stmtname}
            )
        setattr(Context.Rib, gen_context_create.__name__, gen_context_create)
        def_simple_stmt(prefix + "begin", stmtname + "Begin", argtypes)
        def_simple_stmt(prefix + "end", stmtname + "End", [])
    #end def_context_stmts

#begin def_rman_stmt
    if is_block :
        def_context_stmts()
    else :
        def_simple_stmt(methname, stmtname, argtypes)
    #end if
#end def_rman_stmt

vector_arg = [conv_num] * 3
matrix_arg = [conv_num] * 16
for methname, stmtname, argtypes, is_block in \
    (
        ("declare", "Declare", [conv_str, conv_str], False),

        ("", "", [conv_str], True),
        # ("end", "End"),
        ("context", "Context", [conv_int], False),
        ("frame", "Frame", [conv_int], True),
        # ("frame_end", "FrameEnd", [], False),
        ("world", "World", [], True),
        # ("world_end", "WorldEnd"),

        ("format", "Format", [conv_int, conv_int, conv_num], False),
        ("frame_aspect_ratio", "FrameAspectRatio", [conv_num], False),
        ("screen_window", "ScreenWindow", [conv_num, conv_num, conv_num, conv_num], False),
        ("crop_window", "CropWindow", [conv_num, conv_num, conv_num, conv_num], False),
        ("projection", "Projection", [conv_str], False),
        ("clipping", "Clipping", [conv_num, conv_num], False),
        ("clipping_plane", "ClippingPlane", [conv_num, conv_num, conv_num, conv_num, conv_num, conv_num], False),
        ("depth_of_field", "DepthOfField", [conv_num, conv_num, conv_num], False),
        ("shutter", "Shutter", [conv_num, conv_num], False),
        ("pixel_variance", "PixelVariance", [conv_num], False),
        ("pixel_samples", "PixelSamples", [conv_num, conv_num], False),
        ("pixel_filter", "PixelFilter", [conv_str, conv_num, conv_num], False),
        ("exposure", "Exposure", [conv_num, conv_num], False),
        ("imager", "Imager", [conv_str], False),
        ("quantize", "Quantize", [conv_str, conv_int, conv_int, conv_int, conv_num], False),
        # ("display", "Display") handled specially
        ("hider", "Hider", [conv_str], False),
        # ("colour_samples", "ColorSamples") treated specially

        ("option", "Option", [conv_str], False),
        ("attribute", "Attribute", [], True),
        # ("attribute_end", "AttributeEnd"),
        # ("colour", "Color") treated specially
        # ("opacity", "Opacity") treated specially

        ("texture_coordinates", "TextureCoordinates", [conv_num, conv_num, conv_num, conv_num, conv_num, conv_num, conv_num, conv_num], False),
        ("light_source", "LightSource", [conv_str, conv_int], False),
        ("area_light_source", "AreaLightSource", [conv_str, conv_int], False),
        ("illuminate", "Illuminate", [conv_int, conv_bool], False),
        ("surface", "Surface", [conv_str], False),
        ("displacement", "Displacement", [conv_str], False),
        ("atmosphere", "Atmosphere", [conv_str], False),
        ("interior", "Interior", [conv_str], False),
        ("exterior", "Exterior", [conv_str], False),

        ("shading_rate", "ShadingRate", [conv_num], False),
        ("shading_interpolation", "ShadingInterpolation", [conv_str], False),
        ("matte", "Matte", [conv_bool], False),
        ("bound", "Bound", [conv_num, conv_num, conv_num, conv_num, conv_num, conv_num], False),
        ("detail", "Detail", [conv_num, conv_num, conv_num, conv_num, conv_num, conv_num], False),
        ("detail_range", "DetailRange", [conv_num, conv_num, conv_num, conv_num], False),
        ("geometric_approximation", "GeometricApproximation", [conv_str, conv_num], False),
        ("orientation", "Orientation", [conv_str], False),
        ("reverse_orientation", "ReverseOrientation", [], False),
        ("sides", "Sides", [conv_int], False),

        ("identity", "Identity", [], False),
        ("transform", "Transform", matrix_arg, False),
        ("concat_transform", "ConcatTransform", matrix_arg, False),
        ("translate", "Translate", vector_arg, False),
        ("rotate", "Rotate", [conv_num] + vector_arg, False),
        ("scale", "Scale", vector_arg, False),
        ("skew", "Skew", [conv_num, conv_num, conv_num, conv_num, conv_num, conv_num, conv_num], False),
        ("coordinate_system", "CoordinateSystem", [conv_str], False),
        ("coord_sys_transform", "CoordSysTransform", [conv_str], False),
        # TransformPoints probably not useful
        ("transform", "Transform", [], True),
        # ("transform_end", "TransformEnd"),

        ("attribute", "Attribute", [conv_str], False),
        ("attribute", "Attribute", [], True),
        # ("attribute_end", "AttributeEnd"),

        ("polygon", "Polygon", [conv_int], False),
        ("general_polygon", "GeneralPolygon", [conv_int_array], False),
        ("points_polygons", "PointsPolygons", [conv_int_array, conv_int_array], False),
        ("points_general_polygons", "PointsGeneralPolygons", [conv_int_array, conv_int_array, conv_int_array], False),

        # ("basis", "Basis") handled specially
        ("patch", "Patch", [conv_str], False),
        ("patch_mesh", "PatchMesh", [conv_str, conv_int, conv_str, conv_int, conv_int], False),
        ("nu_patch", "NuPatch", [conv_int, conv_int, conv_num_array, conv_num, conv_num, conv_int, conv_int, conv_num_array, conv_num, conv_num], False),
        ("trim_curve", "TrimCurve", [conv_int, conv_int_array, conv_int_array, conv_num_array, conv_num, conv_num, conv_int_array, conv_num_array, conv_num_array, conv_num_array], False),

        ("subdivision_mesh", "SubdivisionMesh", [conv_str, conv_int, conv_int_array, conv_int_array, conv_int, conv_str_array, conv_int_array, conv_int_array, conv_num_array], False),

        ("sphere", "Sphere", [conv_num, conv_num, conv_num, conv_num], False),
        ("cone", "Cone", [conv_num, conv_num, conv_num], False),
        ("cylinder", "Cylinder", [conv_num, conv_num, conv_num, conv_num], False),
        ("hyperboloid", "Hyperboloid", [conv_point, conv_point, conv_num], False),
        ("paraboloid", "Paraboloid", [conv_num, conv_num, conv_num, conv_num], False),
        ("disk", "Disk", [conv_num, conv_num, conv_num], False),
        ("torus", "Torus", [conv_num, conv_num, conv_num, conv_num, conv_num], False),

        ("points", "Points", [conv_int], False),
        ("curves", "Curves", [conv_str, conv_int_array, conv_str], False),
        ("blobby", "Blobby", [conv_int, conv_int, conv_int_array, conv_num, conv_num_array, conv_int, conv_str_array], False),

        # ("procedural", "Procedural") handled specially
        ("geometry", "Geometry", [conv_str], False),

        ("solid", "Solid", [conv_str], True),
        # ("solid_end", "SolidEnd"),
        # ("object_begin", "ObjectBegin") handled specially
        ("object_end", "ObjectEnd", [], False),
        # ("object_instance", "ObjectInstance") handled specially

        # MotionBegin, MotionEnd handled specially

         # todo for all following texture utilities calls: support additional parameters?
        ("make_texture", "MakeTexture", [conv_str, conv_str, conv_str, conv_str, conv_str, conv_num, conv_num], False),
        ("make_lat_long_environment", "MakeLatLongEnvironment", [conv_str, conv_str, conv_str, conv_num, conv_num], False),
        ("make_cube_face_environment", "MakeCubeFaceEnvironment", [conv_str, conv_str, conv_str, conv_str, conv_str, conv_str, conv_str, conv_num, conv_str, conv_num, conv_num], False),
        ("make_shadow", "MakeShadow", [conv_str, conv_str], False),

        ("error_handler", "ErrorHandler", [conv_str], False),
        # ("read_archive", "ReadArchive") handled specially
    ) \
:
    def_rman_stmt(methname, stmtname, argtypes, is_block)
#end for

del methname, stmtname, argtypes, is_block
del vector_arg, matrix_arg

del def_rman_stmt # your work is done
