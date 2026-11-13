# v8 Modern Baihua Source Pool Notes

These sources were prepared after the v7 style-contrast split was already created.
They are not part of v7 training.

## Newly prepared usable sources

### Yu Dafu

- `wikisource_yudafu_chunfengchenzuidewanshang_001`
  - chunks: 7
  - soft-label rows: 18

- `wikisource_yudafu_niaoluoxing_001`
  - soft-label rows: 30

- `wikisource_yudafu_nanqian_001`
  - chunks: 32
  - soft-label rows: 118

### Xu Dishan

- `wikisource_xudishan_mingmingniao_001`
  - soft-label rows: 30

## Existing usable modern sources before this batch

### Lu Xun

- `wikisource_luxun_nahan_001`
- `wikisource_luxun_panghuang_001`
- `wikisource_luxun_gushixinbian_001`

### Xu Dishan

- `wikisource_xudishan_shangrenfu_001`

### Yu Dafu

- `wikisource_yudafu_chenlun_001`

### Xiao Hong

- `wikisource_xiaohong_hulanhezhuan_001`

## Failed or quarantined fetches

- `wikisource_xudishan_haishijian_001`
  - fetched only page navigation chrome
  - quarantined, not usable

- `wikisource_xiaohong_shengsichang_001`
  - fetched only page navigation chrome
  - quarantined, not usable

## Notes

This pool should be used for v8 or later, not retroactively added to v7.

Main future value:

- stronger same-author cross-story negatives for Yu Dafu
- Xu Dishan same-author contrast is now possible
- better modern cross-author style contrast beyond Lu Xun / Xiao Hong

## Additional Zhu Ziqing prose sources

These were prepared while v8 was training and are not part of v8.

Usable:

- `wikisource_zhuziqing_hetangyuese_001`
  - soft-label rows: 1
  - very small, useful only as style/prose reference

- `wikisource_zhuziqing_jiangshengdengyingdeqinhuaihe_001`
  - soft-label rows: 18
  - usable modern prose sample

Failed / quarantined:

- `wikisource_zhuziqing_beiying_001`
  - fetched only Wikisource page chrome, not正文
- `wikisource_zhuziqing_congcong_001`
  - fetched only Wikisource page chrome, not正文

Note: Zhu Ziqing prose is useful for later modern-prose false-friend contrast, but this batch is too small to affect training by itself.

## Lai He follow-up

Purpose: Taiwan Japanese-period / local vernacular source pool. Not used in v8/v9 mainline.

Available:
- wikisource_laihe_yiganchengzi_001
- wikisource_laihe_buruyideguonian_001
- wikisource_laihe_shexiansheng_001

Unavailable / isolated:
- wikisource_laihe_keliantasile_001: Wikisource returned 404 / unavailable during retry.

Decision:
Keep the three available Lai He stories as future Taiwan modern baihua contrast material.
Do not use this pool in v9. v9 remains focused on late-Qing same-book far negatives.
