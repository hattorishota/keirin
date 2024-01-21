<!DOCTYPE html>
<html lang="ja">
  <head>
  <?php
  /**
  * The main template file
  *
  */
    get_header();
  ?>
  </head>
  <body>
    <div class="wrapper">
      <?php get_header('tmp'); ?>
      <main class="main top">
      <?php get_header('list'); ?>
        <section class="fv">
          <div class="fv__inner">
            <div class="fv__slider">
              <div class="fv__item">
                <div class="fv__img">
                  <img src="<?php echo get_template_directory_uri(); ?>/img/top/fv/fv_1_pc.webp" alt="" class="pc-only">
                  <img src="<?php echo get_template_directory_uri(); ?>/img/top/fv/fv_1_sp.webp" alt="" class="sp-only">
                </div>
                <a href="<?php echo home_url('imaginationcontroll'); ?>" class="fv__button">イマジネーションコントロール<br class="sp-only">チラー詳細情報</a>
              </div>
              <div class="fv__item">
                <div class="fv__img">
                  <img src="<?php echo get_template_directory_uri(); ?>/img/top/fv/fv_2_pc.webp" alt="" class="pc-only">
                  <img src="<?php echo get_template_directory_uri(); ?>/img/top/fv/fv_2_sp.webp" alt="" class="sp-only">
                </div>
                <a href="<?php echo home_url('imaginationcontroll'); ?>" class="fv__button">イマジネーションコントロール<br class="sp-only">チラー詳細情報</a>
              </div>
              <div class="fv__item">
                <div class="fv__img">
                  <img src="<?php echo get_template_directory_uri(); ?>/img/top/fv/fv_3_pc.webp" alt="" class="pc-only">
                  <img src="<?php echo get_template_directory_uri(); ?>/img/top/fv/fv_3_sp.webp" alt="" class="sp-only">
                </div>
                <a href="<?php echo home_url('jackettankdedicatedchiller'); ?>" class="fv__button">ジャケットタンク温度制御専用<br class="sp-only">チラー詳細情報</a>
              </div>
            </div>
          </div>
        </section>
        <section class="about">
          <div class="about__inner">
            <div class="section__leading">
              <h2 class="section__leading--ja">About us</h2>
              <span class="section__leading--en">私たちについて</span>
            </div>
            <div class="about__wrap">
              <div class="about__txtBox">
                <p class="about__txt">
                  テキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキスト
                </p>
                <p class="about__txt">
                  テキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキスト
                </p>
                <a href="<?php echo home_url('about'); ?>" class="btn about__button">View More</a>
              </div>
              <div class="about__imgBox">
                <img src="<?php echo get_template_directory_uri(); ?>/img/top/about/about_pc.png" alt="" class="pc-only">
                <img src="<?php echo get_template_directory_uri(); ?>/img/top/about/about_sp.png" alt="" class="sp-only">
              </div>
            </div>
          </div>
        </section>
        <section class="product">
          <div class="product__inner">
            <div class="section__leading">
              <h2 class="section__leading--ja">Product Catalog</h2>
              <span class="section__leading--en">製品カタログ</span>
            </div>
            <div class="product__wrap">
              <ul class="product__list">
                <li class="product__item">
                  <a href="<?php echo home_url('purewater'); ?>">
                    <div class="product__img">
                      <img src="<?php echo get_template_directory_uri(); ?>/img/top/product/straight_one_patisuira.webp" alt="">
                    </div>
                    <div class="product__box">
                      <p class="product__name">純水高精密<br class="sp-only">ワンパスチラー</p>
                      <div class="product__txtBox">
                        <p class="product__txt">世界初！熱交換器がチタン製の特別使用（超純水専用）</p>
                        <p class="product__more"><span>View More</span></p>
                      </div>
                    </div>
                  </a>
                </li>
                <li class="product__item">
                  <a href="<?php echo home_url('ultrapurewater'); ?>">
                    <div class="product__img">
                      <img src="<?php echo get_template_directory_uri(); ?>/img/top/product/straight_one_patisuira.webp" alt="">
                    </div>
                    <div class="product__box">
                      <p class="product__name">超純水高速追従<br class="sp-only">ワンパスチラー</p>
                      <div class="product__txtBox">
                        <p class="product__txt">世界初！熱交換器がチタン製の特別使用（超純水専用）</p>
                        <p class="product__more"><span>View More</span></p>
                      </div>
                    </div>
                  </a>
                </li>
                <li class="product__item">
                  <a href="<?php echo home_url('titaniumheatexchanger'); ?>">
                    <div class="product__img">
                      <img src="<?php echo get_template_directory_uri(); ?>/img/top/product/straight_one_patisuira.webp" alt="">
                    </div>
                    <div class="product__box">
                      <p class="product__name">超純水専用<br class="sp-only">チタン製熱交換機</p>
                      <div class="product__txtBox">
                        <p class="product__txt">世界初！熱交換器がチタン製の特別使用（超純水専用）</p>
                        <p class="product__more"><span>View More</span></p>
                      </div>
                    </div>
                  </a>
                </li>
                <li class="product__item">
                  <a href="<?php echo home_url('highresponsechiller'); ?>">
                    <div class="product__img">
                      <img src="<?php echo get_template_directory_uri(); ?>/img/top/product/straight_one_patisuira.webp" alt="">
                    </div>
                    <div class="product__box">
                      <p class="product__name">ハイレスポンス<br class="sp-only">チラー</p>
                      <div class="product__txtBox">
                        <p class="product__txt">世界初！熱交換器がチタン製の特別使用（超純水専用）</p>
                        <p class="product__more"><span>View More</span></p>
                      </div>
                    </div>
                  </a>
                </li>
                <li class="product__item">
                  <a href="<?php echo home_url('jackettankchiller'); ?>">
                    <div class="product__img">
                      <img src="<?php echo get_template_directory_uri(); ?>/img/top/product/straight_one_patisuira.webp" alt="">
                    </div>
                    <div class="product__box">
                      <p class="product__name">高効率ジャケット<br class="sp-only">タンク＆チラー</p>
                      <div class="product__txtBox">
                        <p class="product__txt">世界初！熱交換器がチタン製の特別使用（超純水専用）</p>
                        <p class="product__more"><span>View More</span></p>
                      </div>
                    </div>
                  </a>
                </li>
                <li class="product__item">
                  <a href="<?php echo home_url('jackettankdedicatedchiller'); ?>">
                    <div class="product__img">
                      <img src="<?php echo get_template_directory_uri(); ?>/img/top/product/straight_one_patisuira.webp" alt="">
                    </div>
                    <div class="product__box">
                      <p class="product__name">ジャケットタンク<br class="sp-only">専用チラー</p>
                      <div class="product__txtBox">
                        <p class="product__txt">世界初！熱交換器がチタン製の特別使用（超純水専用）</p>
                        <p class="product__more"><span>View More</span></p>
                      </div>
                    </div>
                  </a>
                </li>
                <li class="product__item">
                  <a href="<?php echo home_url('coldmachine'); ?>">
                    <div class="product__img">
                      <img src="<?php echo get_template_directory_uri(); ?>/img/top/product/straight_one_patisuira.webp" alt="">
                    </div>
                    <div class="product__box">
                      <p class="product__name">汎用チラー<br class="sp-only">シリーズ</p>
                      <div class="product__txtBox">
                        <p class="product__txt">世界初！熱交換器がチタン製の特別使用（超純水専用）</p>
                        <p class="product__more"><span>View More</span></p>
                      </div>
                    </div>
                  </a>
                </li>
                <li class="product__item">
                  <a href="<?php echo home_url('choicechillercold'); ?>">
                    <div class="product__img">
                      <img src="<?php echo get_template_directory_uri(); ?>/img/top/product/straight_one_patisuira.webp" alt="">
                    </div>
                    <div class="product__box">
                      <p class="product__name">チョイスチラー<br>空冷一体型・水冷一体型</p>
                      <div class="product__txtBox">
                        <p class="product__txt">世界初！熱交換器がチタン製の特別使用（超純水専用）</p>
                        <p class="product__more"><span>View More</span></p>
                      </div>
                    </div>
                  </a>
                </li>
                <li class="product__item">
                  <a href="<?php echo home_url('choicechillerseparate'); ?>">
                    <div class="product__img">
                      <img src="<?php echo get_template_directory_uri(); ?>/img/top/product/straight_one_patisuira.webp" alt="">
                    </div>
                    <div class="product__box">
                      <p class="product__name">チョイスチラー<br class="sp-only">セパレート型</p>
                      <div class="product__txtBox">
                        <p class="product__txt">世界初！熱交換器がチタン製の特別使用（超純水専用）</p>
                        <p class="product__more"><span>View More</span></p>
                      </div>
                    </div>
                  </a>
                </li>
                <li class="product__item">
                  <a href="">
                    <div class="product__img">
                      <img src="<?php echo get_template_directory_uri(); ?>/img/top/product/straight_one_patisuira.webp" alt="">
                    </div>
                    <div class="product__box">
                      <p class="product__name">イマジネーション<br class="sp-only">コントロールチラー</p>
                      <div class="product__txtBox">
                        <p class="product__txt">世界初！熱交換器がチタン製の特別使用（超純水専用）</p>
                        <p class="product__more"><span>View More</span></p>
                      </div>
                    </div>
                  </a>
                </li>
                <li class="product__item">
                  <a href="">
                    <div class="product__img">
                      <img src="<?php echo get_template_directory_uri(); ?>/img/top/product/straight_one_patisuira.webp" alt="">
                    </div>
                    <div class="product__box">
                      <p class="product__name">小型卓上空冷チラー</p>
                      <div class="product__txtBox">
                        <p class="product__txt">世界初！熱交換器がチタン製の特別使用（超純水専用）</p>
                        <p class="product__more"><span>View More</span></p>
                      </div>
                    </div>
                  </a>
                </li>
                <li class="product__item">
                  <a href="">
                    <div class="product__img">
                      <img src="<?php echo get_template_directory_uri(); ?>/img/top/product/straight_one_patisuira.webp" alt="">
                    </div>
                    <div class="product__box">
                      <p class="product__name">小型精密チラー</p>
                      <div class="product__txtBox">
                        <p class="product__txt">世界初！熱交換器がチタン製の特別使用（超純水専用）</p>
                        <p class="product__more"><span>View More</span></p>
                      </div>
                    </div>
                  </a>
                </li>
                <li class="product__item">
                  <a href="">
                    <div class="product__img">
                      <img src="<?php echo get_template_directory_uri(); ?>/img/top/product/straight_one_patisuira.webp" alt="">
                    </div>
                    <div class="product__box">
                      <p class="product__name">特注品</p>
                      <div class="product__txtBox">
                        <p class="product__txt">世界初！熱交換器がチタン製の特別使用（超純水専用）</p>
                        <p class="product__more"><span>View More</span></p>
                      </div>
                    </div>
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </section>
        <section class="introduce">
          <div class="introduce__inner">
            <div class="introduce__txtBox">
              <p class="introduce__leading">テキストテキストテキステキストテキストテキスト</p>
              <p class="introduce__txt">テキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキストテキスト</p>
            </div>
            <div class="introduce__movie">
              <iframe id="iframe-1" width="653" height="381" src="https://www.youtube.com/embed/sZ8dbDAIPUM?si=5DLTLpbYoEtTCCM1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
            </div>
          </div>
        </section>
        <section class="news">
          <div class="news__inner">
            <div class="news__img">
              <img src="<?php echo get_template_directory_uri(); ?>/img/top/news/kv_pc.webp" alt="" class="pc-only">
              <img src="<?php echo get_template_directory_uri(); ?>/img/top/news/kv_sp.webp" alt="" class="sp-only">
            </div>
            <div class="news__wrap">
              <div class="section__leading">
                <h2 class="section__leading--ja">News & Blog</h2>
                <span class="section__leading--en">お知らせ ＆ ブログ</span>
              </div>
              <ul class="news__list">
                <li class="news__item">
                  <div class="news__box">
                    <span class="news__date">2023/08/31</span>
                    <span class="news__cat news__cat--news">News</span>
                  </div>
                  <a href="" class="news__title">サイトをリニューアルしました</a>
                </li>
                <li class="news__item">
                  <div class="news__box">
                    <span class="news__date">2023/08/31</span>
                    <span class="news__cat news__cat--blog">Blog</span>
                  </div>
                  <a href="" class="news__title">サイトをリニューアルしました</a>
                </li>
                <li class="news__item">
                  <div class="news__box">
                    <span class="news__date">2023/08/31</span>
                    <span class="news__cat news__cat--news">News</span>
                  </div>
                  <a href="" class="news__title">サイトをリニューアルしました</a>
                </li>
                <li class="news__item">
                  <div class="news__box">
                    <span class="news__date">2023/08/31</span>
                    <span class="news__cat news__cat--blog">Blog</span>
                  </div>
                  <a href="" class="news__title">サイトをリニューアルしました</a>
                </li>
                <li class="news__item">
                  <div class="news__box">
                    <span class="news__date">2023/08/31</span>
                    <span class="news__cat news__cat--news">News</span>
                  </div>
                  <a href="" class="news__title">サイトをリニューアルしました</a>
                </li>
              </ul>
              <div class="news__buttonBox">
                <a href="<?php echo home_url('news'); ?>" class="btn">View More</a>
              </div>
            </div>
          </div>
        </section>
        <?php get_footer(); ?>