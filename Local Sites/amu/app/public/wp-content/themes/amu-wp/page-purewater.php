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
      <main class="main under">
      <?php get_header('list'); ?>
        <section class="section product">
            <div class="product__fv">
                <picture>
                    <source media="(max-width: 870px)" srcset="<?php echo wp_get_attachment_url(SCF::get('fv_pc')); ?>">
                    <img src="<?php echo wp_get_attachment_url(SCF::get('fv_pc')); ?>" alt="">
                </picture>
            </div>
            <div class="product__inner">
                <h2 class="product__leading"><?php echo SCF::get('title'); ?></h2>
                <div class="product__wrap">
                    <div class="product__featureBox">
                        <h3 class="product__featureLeading"><?php echo SCF::get('leading'); ?></h3>
                        <div class="product__featureInner">
                            <div class="product__featureFlex">
                                <div class="product__featureTxtBox">
                                    <p class="product__featureTxt">
                                    <?php echo SCF::get('content'); ?>
                                    </p>
                                </div>
                                <div class="product__featureImgBox">
                                    <img src="<?php echo wp_get_attachment_url(SCF::get('image')); ?>" alt="">
                                </div>
                            </div>
                            <div class="product__featureKv">
                                <picture>
                                    <source media="(max-width: 870px)" srcset="<?php echo wp_get_attachment_url(SCF::get('image_big')); ?>">
                                    <img src="<?php echo wp_get_attachment_url(SCF::get('image_big')); ?>" alt="">
                                </picture>
                            </div>
                        </div>
                    </div>
                    <ul class="product__list">
                    <?php
                        $product_list = SCF::get('product_list');
                        foreach ($product_list as $fields) { 
                    ?>
                        <li class="product__item">
                            <div class="product__labelBox">
                                <div class="product__nameBox">
                                    <p class="product__name">純粋高精密ワンパスチラー</p>
                                    <p class="product__number"><?php echo $fields['product_number']; ?></p>
                                </div>
                                <div class="product__linkBox">
                                    <p class="product__price">¥<?php echo $fields['product_price']; ?></p>
                                    <a href="" class="product__link">商品カタログはこちら</a>
                                </div>
                            </div>
                            <div class="product__detailBox">
                                <div class="product__detailFlex">
                                    <div class="product__detailImg">
                                        <img src="<?php echo wp_get_attachment_url($fields['product_image']); ?>" alt="">
                                    </div>
                                    <div class="product__detailTxtBox">
                                        <table class="product__detailTable">
                                            <tr>
                                                <th>温度範囲（℃）</th>
                                                <th>温度安定度</th>
                                                <th>冷却能力/加熱能力</th>
                                                <th>処理流量/温度差</th>
                                            </tr>
                                            <tr>
                                                <td><?php echo $fields['tempreture_range']; ?></td>
                                                <td><?php echo $fields['degree_stabled_rate']; ?></td>
                                                <td><?php echo $fields['product_ability']; ?></td>
                                                <td><?php echo $fields['tempreture_gap']; ?></td>
                                            </tr>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </li>
                    <?php } ?>
                    </ul>
                    <div class="product__banner">
                        <a href="">
                            <img src="<?php echo wp_get_attachment_url(SCF::get('button_image')); ?>" alt="">
                        </a>
                    </div>
                </div>
            </div>
        </section>
        <?php get_footer(); ?>