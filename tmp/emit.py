import argparse
from itertools import product
from string import Template

np_to_cpp_map = {
    'bool': 'bool',
    'int8': 'int8_t',
    'int16': 'int16_t',
    'int32': 'int32_t',
    'int64': 'int64_t',

    'uint8': 'uint8_t',
    'uint16': 'uint16_t',
    'uint32': 'uint32_t',
    'uint64': 'uint64_t',

    'float32': 'float',
    'float64': 'double',
    'complex64': 'std::complex<float>',
    'complex128': 'std::complex<double>'
}


kernel_fill_template = Template('''template <>
    ERROR NumpyArray_fill<${FROM}, ${TO}>(
        kernel::lib ptr_lib,
        ${TO} *toptr,
        int64_t tooffset,
        const ${FROM} *fromptr,
        int64_t length) {
        if (ptr_lib == kernel::lib::cpu) {
            return awkward_NumpyArray_fill_to${NPTO}_from${NPFROM}(
                toptr,
                tooffset,
                fromptr,
                length);
        }
        else if (ptr_lib == kernel::lib::cuda) {
            throw std::runtime_error(
                std::string("not implemented: ptr_lib == cuda_kernels "
                            "for NumpyArray_fill<${TO}, ${FROM}>"
                            + FILENAME(__LINE__)));
        }
        else {
            throw std::runtime_error(
                std::string("unrecognized ptr_lib "
                            "for NumpyArray_fill<${TO}, ${FROM}>"
                            + FILENAME(__LINE__)));
        }
    }''')


oph_fill_template = Template('''/// @param toptr outparam
  /// @param tooffset inparam role: IndexedArray-index-offset
  /// @param fromptr inparam role: NumpyArray-ptr
  /// @param length inparam
  EXPORT_SYMBOL struct Error
  awkward_NumpyArray_fill_to${NPTO}_from${NPFROM}(
    ${TO} * toptr,
    int64_t tooffset,
    const ${FROM} * fromptr,
    int64_t length);''')

opcpp_file_template = Template('''    ERROR
    awkward_NumpyArray_fill_to${NPTO}_from${NPFROM}(${TO}* toptr,
                                            int64_t tooffset,
                                            const ${FROM}* fromptr,
                                            int64_t length) {
    return awkward_NumpyArray_fill(toptr, tooffset, fromptr, length);
    }''')


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("file", choices=["oph", "opcpp", "kernel"])
    return p


if __name__ == "__main__":
    args = create_parser().parse_args()

    from_to = list(product(np_to_cpp_map.keys(), np_to_cpp_map.keys()))

    for np_from_type, np_to_type in from_to:
        cpp_from_type = np_to_cpp_map[np_from_type]
        cpp_to_type = np_to_cpp_map[np_to_type]

        if args.file == "oph":
            template = oph_fill_template
        elif args.file == "opcpp":
            template = opcpp_file_template
        elif args.file == "kernel":
            template = kernel_fill_template
        else:
            raise ValueError(f"Invalid output file {args.file}")

        fn = template.substitute(TO=cpp_to_type, FROM=cpp_from_type,
                                 NPTO=np_to_type, NPFROM=np_from_type)



        print(fn)
