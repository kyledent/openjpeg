# Tests
include_directories(
  ${OPENJPEG_BINARY_DIR}/src/lib/openjp2 # opj_config.h
  ${OPENJPEG_BINARY_DIR}/src/bin/common # opj_apps_config.h
  ${OPENJPEG_SOURCE_DIR}/src/lib/openjp2
  ${OPENJPEG_SOURCE_DIR}/src/bin/jp2
  ${OPENJPEG_SOURCE_DIR}/src/bin/common
  ${Z_INCLUDE_DIRNAME}
  ${PNG_INCLUDE_DIRNAME}
  ${TIFF_INCLUDE_DIRNAME}
  )

# First thing define the common source:
set(compare_images_SRCS compare_images.c
  ${OPENJPEG_SOURCE_DIR}/src/bin/jp2/convert.c
  ${OPENJPEG_SOURCE_DIR}/src/bin/jp2/converttif.c
  ${OPENJPEG_SOURCE_DIR}/src/bin/common/opj_getopt.c
  )

set(compare_dump_files_SRCS compare_dump_files.c
  ${OPENJPEG_SOURCE_DIR}/src/bin/common/opj_getopt.c)

set(compare_raw_files_SRCS compare_raw_files.c
  ${OPENJPEG_SOURCE_DIR}/src/bin/common/opj_getopt.c)

add_executable(compare_images ${compare_images_SRCS})
target_link_libraries(compare_images
  ${OPENJPEG_LIBRARY_NAME}
  ${PNG_LIBNAME} ${TIFF_LIBNAME}
  )
# To support universal exe:
if(ZLIB_FOUND AND APPLE)
  target_link_libraries(compare_images z)
else(ZLIB_FOUND AND APPLE)
  target_link_libraries(compare_images ${Z_LIBNAME})
endif()

add_executable(compare_dump_files ${compare_dump_files_SRCS})

add_executable(j2k_random_tile_access j2k_random_tile_access.c)
target_link_libraries(j2k_random_tile_access ${OPENJPEG_LIBRARY_NAME})

add_executable(test_tile_part_interleaved test_tile_part_interleaved.c)
target_link_libraries(test_tile_part_interleaved ${OPENJPEG_LIBRARY_NAME})

add_executable(compare_raw_files ${compare_raw_files_SRCS})

add_executable(test_tile_encoder test_tile_encoder.c)
target_link_libraries(test_tile_encoder ${OPENJPEG_LIBRARY_NAME})

add_executable(test_decode_area test_decode_area.c)
target_link_libraries(test_decode_area ${OPENJPEG_LIBRARY_NAME})

# In-memory lossless round trips: every code-block style switch, HT (64) and
# HT with VSC (72) included, signed and unsigned samples, and the inline (0)
# and one-worker (1) thread paths. Every run also checks the Rsiz bit and the
# CAP marker (and, for JP2, the brand) against the code-block style.
add_executable(test_encode_roundtrip test_encode_roundtrip.c)
target_link_libraries(test_encode_roundtrip ${OPENJPEG_LIBRARY_NAME})
foreach(mode 0 1 2 4 5 8 9 16 32 63 64 72)
  add_test(NAME roundtrip_mode_${mode} COMMAND test_encode_roundtrip -mode ${mode})
  add_test(NAME roundtrip_mode_${mode}_s16 COMMAND test_encode_roundtrip -mode ${mode} -prec 16 -sgnd 1)
endforeach()
foreach(t 0 1 2 4)
  add_test(NAME roundtrip_threads_${t} COMMAND test_encode_roundtrip -threads ${t} -dec-threads ${t} -w 1030 -h 517)
endforeach()
add_test(NAME roundtrip_u8_small COMMAND test_encode_roundtrip -prec 8 -w 17 -h 5 -res 2)
# Selective arithmetic coding bypass: raw passes, including code-blocks whose
# height is not a multiple of 4 (the partial last stripe).
foreach(mode 1 5 9 17)
  add_test(NAME roundtrip_bypass_${mode}_odd COMMAND test_encode_roundtrip -mode ${mode} -w 203 -h 201 -cblk 32 -prec 16 -sgnd 1)
  add_test(NAME roundtrip_bypass_${mode}_tall COMMAND test_encode_roundtrip -mode ${mode} -w 1030 -h 517 -prec 14)
endforeach()
add_test(NAME roundtrip_jp2 COMMAND test_encode_roundtrip -jp2)
# Thread pools reused across codecs: alternating directions, mixed sizes,
# no accumulation of threads, and fork() safety (POSIX only).
if(UNIX)
  add_executable(test_thread_pool_reuse test_thread_pool_reuse.c)
  target_link_libraries(test_thread_pool_reuse ${OPENJPEG_LIBRARY_NAME})
  add_test(NAME thread_pool_reuse COMMAND test_thread_pool_reuse)
  add_test(NAME thread_pool_reuse_disabled COMMAND test_thread_pool_reuse)
  set_tests_properties(thread_pool_reuse_disabled PROPERTIES ENVIRONMENT "OPJ_THREAD_POOL_CACHE=0")
endif()
# HT code-blocks: precisions, image sizes that are not multiples of the
# code-block size, code-block shapes, resolution counts, threads, JP2 (brand
# "jph "), several components, tiles, precincts, and the irreversible
# transform (whose decoded image must equal the MQ-coded one).
add_test(NAME roundtrip_ht_u8 COMMAND test_encode_roundtrip -mode 64 -prec 8)
add_test(NAME roundtrip_ht_u16 COMMAND test_encode_roundtrip -mode 64 -prec 16)
add_test(NAME roundtrip_ht_s16_1030x517 COMMAND test_encode_roundtrip -mode 64 -prec 16 -sgnd 1 -w 1030 -h 517)
add_test(NAME roundtrip_ht_u8_small COMMAND test_encode_roundtrip -mode 64 -prec 8 -w 17 -h 5 -res 2)
foreach(cb 64x64 32x32 16x16 64x32 32x64 128x16 1024x4 4x1024)
  string(REPLACE "x" ";" cbwh ${cb})
  list(GET cbwh 0 cbw)
  list(GET cbwh 1 cbh)
  add_test(NAME roundtrip_ht_cblk_${cb} COMMAND test_encode_roundtrip -mode 64 -cblkw ${cbw} -cblkh ${cbh} -w 1030 -h 1100)
endforeach()
foreach(res 1 2 3 4 5 6)
  add_test(NAME roundtrip_ht_res_${res} COMMAND test_encode_roundtrip -mode 64 -res ${res})
endforeach()
foreach(t 0 1 4)
  add_test(NAME roundtrip_ht_threads_${t} COMMAND test_encode_roundtrip -mode 64 -threads ${t} -dec-threads ${t} -w 1030 -h 517)
endforeach()
add_test(NAME roundtrip_ht_jp2 COMMAND test_encode_roundtrip -mode 64 -jp2)
add_test(NAME roundtrip_ht_vsc_jp2 COMMAND test_encode_roundtrip -mode 72 -jp2 -prec 16 -sgnd 1)
add_test(NAME roundtrip_ht_rgb COMMAND test_encode_roundtrip -mode 64 -comps 3 -mct 1 -prec 8)
add_test(NAME roundtrip_ht_tiles COMMAND test_encode_roundtrip -mode 64 -tile 128 -res 4)
add_test(NAME roundtrip_ht_precincts COMMAND test_encode_roundtrip -mode 64 -prc 64 -cblk 32)
add_test(NAME roundtrip_ht_irrev COMMAND test_encode_roundtrip -mode 64 -irrev)
add_test(NAME roundtrip_ht_irrev_rgb_s16 COMMAND test_encode_roundtrip -mode 64 -irrev -comps 3 -mct 1 -prec 16 -sgnd 1)
# Code-block style bits the encoder does not implement must be refused, not
# written into COD over code-blocks coded otherwise.
foreach(mode 0x80 0xC0 0x100)
  add_test(NAME reject_mode_${mode} COMMAND test_encode_roundtrip -mode ${mode} -expect-reject)
endforeach()
# HT code-blocks: Part-1 mode switches other than VSC do not apply; a single
# cleanup pass cannot meet a rate target or fill several layers; no ROI; at
# most 30 magnitude bit-planes (checked after the guard bits are final).
foreach(mode 0x41 0x42 0x44 0x50 0x60)
  add_test(NAME reject_ht_mode_${mode} COMMAND test_encode_roundtrip -mode ${mode} -expect-reject)
endforeach()
add_test(NAME reject_ht_rate COMMAND test_encode_roundtrip -mode 64 -rate 20 -expect-reject)
add_test(NAME reject_ht_layers COMMAND test_encode_roundtrip -mode 64 -layers 2 -expect-reject)
add_test(NAME reject_ht_roi COMMAND test_encode_roundtrip -mode 64 -roi -expect-reject)
add_test(NAME reject_ht_prec28 COMMAND test_encode_roundtrip -mode 64 -prec 28 -expect-reject)
add_test(NAME reject_ht_prec26_guard4 COMMAND test_encode_roundtrip -mode 64 -prec 26 -guard 4 -expect-reject)
# ... and a coefficient beyond the range of its subband (no guard bits) is
# refused rather than written as an undecodable code-block
add_test(NAME reject_ht_guard0_checker COMMAND test_encode_roundtrip -mode 64 -prec 8 -w 256 -h 256 -checker 2 -guard 0 -expect-reject)
add_test(NAME roundtrip_ht_guard1_checker COMMAND test_encode_roundtrip -mode 64 -prec 8 -w 256 -h 256 -checker 2 -guard 1)

# HT block decoder: the scalar and the default (SIMD when the CPU supports
# it) implementations must agree on random code-blocks. The test program
# uses internal functions of the library and is built with the unit tests.
if(BUILD_UNIT_TESTS AND UNIX)
  add_test(NAME ht_dec_scalar_vs_default COMMAND test_ht_dec 20000)
endif()

# Let's try a couple of possibilities:
add_test(NAME tte0 COMMAND test_tile_encoder)
add_test(NAME tte1 COMMAND test_tile_encoder 3 2048 2048 1024 1024 8 1 tte1.j2k)
add_test(NAME tte2 COMMAND test_tile_encoder 3 2048 2048 1024 1024 8 1 tte2.jp2)
add_test(NAME tte3 COMMAND test_tile_encoder 1 2048 2048 1024 1024 8 1 tte3.j2k)
add_test(NAME tte4 COMMAND test_tile_encoder 1  256  256  128  128 8 0 tte4.j2k)
add_test(NAME tte5 COMMAND test_tile_encoder 1  512  512  256  256 8 0 tte5.j2k)
#add_test(NAME tte6 COMMAND test_tile_encoder 1 8192 8192  512  512 8 0 tte6.j2k)
#add_test(NAME tte7 COMMAND test_tile_encoder 1 32768 32768 512  512 8 0 tte7.jp2)

add_executable(test_tile_decoder test_tile_decoder.c)
target_link_libraries(test_tile_decoder ${OPENJPEG_LIBRARY_NAME})

add_test(NAME ttd0 COMMAND test_tile_decoder)
set_property(TEST ttd0 APPEND PROPERTY DEPENDS tte0)
add_test(NAME ttd1 COMMAND test_tile_decoder 0 0 1024 1024 tte1.j2k)
set_property(TEST ttd1 APPEND PROPERTY DEPENDS tte1)
add_test(NAME ttd2 COMMAND test_tile_decoder 0 0 1024 1024 tte2.jp2)
set_property(TEST ttd2 APPEND PROPERTY DEPENDS tte2)
#add_test(NAME ttd6 COMMAND test_tile_decoder 0 0  512  512 tte6.j2k)
#set_property(TEST ttd6 APPEND PROPERTY DEPENDS tte6)
#add_test(NAME ttd7 COMMAND test_tile_decoder 0 0  512  512 tte7.jp2)
#set_property(TEST ttd7 APPEND PROPERTY DEPENDS tte7)

add_test(NAME rta1 COMMAND j2k_random_tile_access tte1.j2k)
set_property(TEST rta1 APPEND PROPERTY DEPENDS tte1)
add_test(NAME rta2 COMMAND j2k_random_tile_access tte2.jp2)
set_property(TEST rta2 APPEND PROPERTY DEPENDS tte2)
add_test(NAME rta3 COMMAND j2k_random_tile_access tte3.j2k)
set_property(TEST rta3 APPEND PROPERTY DEPENDS tte3)
add_test(NAME rta4 COMMAND j2k_random_tile_access tte4.j2k)
set_property(TEST rta4 APPEND PROPERTY DEPENDS tte4)
add_test(NAME rta5 COMMAND j2k_random_tile_access tte5.j2k)
set_property(TEST rta5 APPEND PROPERTY DEPENDS tte5)

# Non-regression test for issue #1558: tile-by-tile decoding of a codestream
# whose tile-parts are interleaved across tiles (self-contained, no test data).
add_test(NAME rta_interleaved_tile_parts COMMAND test_tile_part_interleaved)

add_test(NAME tda_prep_reversible_no_precinct COMMAND test_tile_encoder 1 256 256 32 32 8 0 reversible_no_precinct.j2k 4 4 3 0 0 1)
add_test(NAME tda_reversible_no_precinct COMMAND test_decode_area -q reversible_no_precinct.j2k)
set_property(TEST tda_reversible_no_precinct APPEND PROPERTY DEPENDS tda_prep_reversible_no_precinct)

add_test(NAME tda_prep_reversible_203_201_17_19_no_precinct COMMAND test_tile_encoder 1 203 201 17 19 8 0 reversible_203_201_17_19_no_precinct.j2k 4 4 3 0 0 1)
add_test(NAME tda_reversible_203_201_17_19_no_precinct COMMAND test_decode_area -q reversible_203_201_17_19_no_precinct.j2k)
set_property(TEST tda_reversible_203_201_17_19_no_precinct APPEND PROPERTY DEPENDS tda_prep_reversible_203_201_17_19_no_precinct)

add_test(NAME tda_prep_reversible_with_precinct COMMAND test_tile_encoder 1 256 256 32 32 8 0 reversible_with_precinct.j2k 4 4 3 0 0 1 16 16)
add_test(NAME tda_reversible_with_precinct COMMAND test_decode_area -q reversible_with_precinct.j2k)
set_property(TEST tda_reversible_with_precinct APPEND PROPERTY DEPENDS tda_prep_reversible_with_precinct)

add_test(NAME tda_prep_irreversible_no_precinct COMMAND test_tile_encoder 1 256 256 32 32 8 1 irreversible_no_precinct.j2k 4 4 3 0 0 1)
add_test(NAME tda_irreversible_no_precinct COMMAND test_decode_area -q irreversible_no_precinct.j2k)
set_property(TEST tda_irreversible_no_precinct APPEND PROPERTY DEPENDS tda_prep_irreversible_no_precinct)

add_test(NAME tda_prep_irreversible_203_201_17_19_no_precinct COMMAND test_tile_encoder 1 203 201 17 19 8 1 irreversible_203_201_17_19_no_precinct.j2k 4 4 3 0 0 1)
add_test(NAME tda_irreversible_203_201_17_19_no_precinct COMMAND test_decode_area -q irreversible_203_201_17_19_no_precinct.j2k)
set_property(TEST tda_irreversible_203_201_17_19_no_precinct APPEND PROPERTY DEPENDS tda_prep_irreversible_203_201_17_19_no_precinct)

add_test(NAME tda_prep_strip COMMAND test_tile_encoder 1 256 256 256 256 8 0 tda_single_tile.j2k)
add_test(NAME tda_strip COMMAND test_decode_area -q -strip_height 3 -strip_check tda_single_tile.j2k)
set_property(TEST tda_strip APPEND PROPERTY DEPENDS tda_prep_strip)

add_executable(include_openjpeg include_openjpeg.c)

# No image send to the dashboard if lib PNG is not available.
if(NOT OPJ_HAVE_LIBPNG)
  message(WARNING "Lib PNG seems to be not available: if you want run the non-regression tests with images reported to the dashboard, you need it (try BUILD_THIRDPARTY)")
endif()

add_subdirectory(conformance)
add_subdirectory(nonregression)
add_subdirectory(unit)

if(BUILD_JPIP)
  if(JPIP_SERVER)
    #set(s "http://jpip.example.com/myFCGI?target=16.jp2&fsiz=170,170&cnew=http&type=jpp-stream")
    set(s "${JPIP_SERVER}?target=16.jp2&fsiz=170,170&cnew=http&type=jpp-stream")
    set(p "${CMAKE_CURRENT_BINARY_DIR}/jpip.dat")
    set(md5 "d41d8cd98f00b204e9800998ecf8427e")
    add_test(NAME TestJPIP1 COMMAND ${CMAKE_COMMAND} -DD_URL:STRING=${s} -DD_FILE:PATH=${p}
      -DEXPECTED_MD5=${md5} -P ${PROJECT_SOURCE_DIR}/cmake/JPIPTestDriver.cmake)
  endif()
endif()

add_executable(ppm2rgb3 ppm2rgb3.c)
#add_executable(pdf2jp2 pdf2jp2.c)
