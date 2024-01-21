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
      <section class="section business">
        <div class="leading__box">
          <h2 class="leading__en">Our Business</h2>
          <p class="leading__jp">事業案内</p>
        </div>
        <div class="fv">
          <div class="fv__inner">
            <div class="fv__item">
              <div class="fv__img">
                <picture>
                  <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/fv_sp.webp">
                  <img src="<?php echo get_template_directory_uri(); ?>/img/business/fv_pc.webp" alt="">
                </picture>
              </div>
              <a href="" class="fv__button">お問い合わせはこちら</a>
            </div>
          </div>
        </div>
        <div class="business__inner">
          <div class="business__wrap business__controller">
            <h2 class="business__leading">コントローラー</h2>
            <picture class="business__controllerImg">
              <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/controller_pc.webp">
              <img src="<?php echo get_template_directory_uri(); ?>/img/business/controller_pc.webp" alt="">
            </picture>
            <ul class="business__list">
              <li class="business__item">専用コントローラーによる精密温度制御を可能にしました。</li>
              <li class="business__item">
                弊社標準製品殆どのラインナップにおいて 格段に高い安定度を実現！<br>
                （標準で±0.1℃　実力値±0.03℃）
              </li>
              <li class="business__item">
                従来の方式で使用していたヒーターをなくしたため、55％以上の省エネ効果！
              </li>
            </ul>
          </div>
          <div class="business__wrap business__system">
            <h2 class="business__leading">3タイプのシステム構造</h2>
            <ul class="business__list">
              <li class="business__item">それぞれ水槽無しタイプの製造も可能です。</li>
              <li class="business__item">屋外設置タイプの製造も可能です。（オプションにより設定）</li>
              <li class="business__item">特注オーダーメイド製品も1台よりお受けいたします。</li>
              <li class="business__item">オゾン層を破壊しないHFC新冷媒であるR407C、404A、134aを採用</li>
            </ul>
          </div>
          <div class="business__imgBox">
            <picture class="business__img">
              <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/air-cold_sp.webp">
              <img src="<?php echo get_template_directory_uri(); ?>/img/business/air-cold.webp" alt="">
            </picture>
            <picture class="business__img">
              <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/water-cold_sp.webp">
              <img src="<?php echo get_template_directory_uri(); ?>/img/business/water-cold.webp" alt="">
            </picture>
            <picture class="business__img">
              <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/separate_sp.webp">
              <img src="<?php echo get_template_directory_uri(); ?>/img/business/separate.webp" alt="">
            </picture>
          </div>
          <div class="business__wrap business__special">
            <h2 class="business__leading">特殊ニーズへの対応</h2>
            <ul class="business__specialList">
              <li class="business__specialItem">
                <picture class="business__img">
                  <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/stable-machine_sp.webp">
                  <img src="<?php echo get_template_directory_uri(); ?>/img/business/stable-machine.webp" alt="">
                </picture>
              </li>
              <li class="business__specialItem">
                <picture class="business__img">
                  <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/build-in-cold-machine_sp.webp">
                  <img src="<?php echo get_template_directory_uri(); ?>/img/business/build-in-cold-machine.webp" alt="">
                </picture>
              </li>
              <li class="business__specialItem">
                <picture class="business__img">
                  <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/templeture-examination-machine_sp.webp">
                  <img src="<?php echo get_template_directory_uri(); ?>/img/business/templeture-examination-machine.webp" alt="">
                </picture>
              </li>
            </ul>
          </div>
          <div class="business__wrap business__support">
            <h2 class="business__leading">サポート・アフターサービス</h2>
            <p class="business__supportLabel">株式会社AMU冷熱は協力業者とネットワークを持ち日本全国～海外まで迅速な製造販売を対応可能にしています。</p>
            <div class="business__supportMap">
              <img src="<?php echo get_template_directory_uri(); ?>/img/business/map.webp" alt="">
            </div>
            <div class="business__tab">
              <div class="business__tabBox">
                <ul class="business__tabList">
                  <li class="business__tabItem current">日本</li>
                  <li class="business__tabItem">アメリカ</li>
                  <li class="business__tabItem">EU諸国</li>
                </ul>
                <p class="business__tabTxt">※エリアをクリックで項目が出現</p>
              </div>
              <div class="business__tabContainer">
                <div class="business__tabContent show">
                  <p class="business__achievementItem--ja">日本全土</p>
                  <ul class="business__achievementList">
                    <li class="business__achievementItem">埼玉</li>
                    <li class="business__achievementItem">東京</li>
                    <li class="business__achievementItem">大阪</li>
                    <li class="business__achievementItem">帯広</li>
                    <li class="business__achievementItem">新潟</li>
                    <li class="business__achievementItem">広島</li>
                    <li class="business__achievementItem">高知</li>
                    <li class="business__achievementItem">北九州</li>
                    <li class="business__achievementItem">広島</li>
                  </ul>
                </div>
                <div class="business__tabContent">
                  <ul class="business__achievementList">
                    <li class="business__achievementItem">カリフォルニア</li>
                    <li class="business__achievementItem">テキサス</li>
                    <li class="business__achievementItem">アリゾナ</li>
                    <li class="business__achievementItem">ニューメキシコ</li>
                    <li class="business__achievementItem">ユタ</li>
                    <li class="business__achievementItem">バージニア</li>
                    <li class="business__achievementItem">バーモント</li>
                    <li class="business__achievementItem">ニューハンプシャー</li>
                    <li class="business__achievementItem">ニューヨーク</li>
                    <li class="business__achievementItem">マンハッタン</li>
                    <li class="business__achievementItem">コネチカット</li>
                    <li class="business__achievementItem">ニュージャージー</li>
                    <li class="business__achievementItem">コロラド</li>
                    <li class="business__achievementItem">ワシントン</li>
                    <li class="business__achievementItem">オレゴン</li>
                    <li class="business__achievementItem">アイダホ</li>
                  </ul>
                </div>
                <div class="business__tabContent">
                  <ul class="business__achievementList">
                    <li class="business__achievementItem">イギリス</li>
                    <li class="business__achievementItem">アイルランド</li>
                    <li class="business__achievementItem">フランス</li>
                    <li class="business__achievementItem">イタリア</li>
                    <li class="business__achievementItem">ドイツ</li>
                    <li class="business__achievementItem">イスラエル</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
          <div class="business__wrap business__achievement">
            <h2 class="business__leading">納品実績</h2>
            <ul class="business__achievementList">
              <li class="business__achievementItem">東京エレクトロン㈱</li>
              <li class="business__achievementItem">㈱東京精密</li>
              <li class="business__achievementItem">ToTo㈱</li>
              <li class="business__achievementItem">㈱湖池屋</li>
              <li class="business__achievementItem">トヨタ自動車㈱</li>
              <li class="business__achievementItem">本田技研工業㈱</li>
              <li class="business__achievementItem">日産自動車㈱</li>
              <li class="business__achievementItem">いすゞ自動車㈱</li>
              <li class="business__achievementItem">スズキ㈱</li>
              <li class="business__achievementItem">アステラス製薬㈱</li>
              <li class="business__achievementItem">富士通㈱</li>
              <li class="business__achievementItem">富士電機㈱</li>
              <li class="business__achievementItem">㈱東芝</li>
              <li class="business__achievementItem">三菱電機㈱</li>
              <li class="business__achievementItem">YAC</li>
              <li class="business__achievementItem">東京大学</li>
              <li class="business__achievementItem">㈱東陽テクニカ</li>
              <li class="business__achievementItem">岩崎電機㈱</li>
              <li class="business__achievementItem">京都大学</li>
              <li class="business__achievementItem">ウシオ電機㈱</li>
              <li class="business__achievementItem">㈱ニシヤマ</li>
              <li class="business__achievementItem">埼玉大学</li>
              <li class="business__achievementItem">㈱島津製作所</li>
              <li class="business__achievementItem">長田電気工業㈱</li>
              <li class="business__achievementItem">慶応義塾大学</li>
              <li class="business__achievementItem">メック㈱</li>
              <li class="business__achievementItem">㈱長田中央研究所</li>
              <li class="business__achievementItem">東京理科大学</li>
              <li class="business__achievementItem">㈲イーオーアール</li>
              <li class="business__achievementItem">双日マシナリー㈱</li>
              <li class="business__achievementItem">東京工芸大学</li>
              <li class="business__achievementItem">昭和炭酸㈱</li>
              <li class="business__achievementItem">入江㈱</li>
              <li class="business__achievementItem">富士通㈱</li>
              <li class="business__achievementItem">カンタムエレクトロニクス㈱</li>
              <li class="business__achievementItem">三和ハイドロテック㈱</li>
              <li class="business__achievementItem">旭硝子㈱</li>
              <li class="business__achievementItem">㈱ニコン</li>
              <li class="business__achievementItem">スカイサイエンス㈱</li>
              <li class="business__achievementItem">㈱高岡</li>
              <li class="business__achievementItem">日本自動車研究所</li>
              <li class="business__achievementItem">日立デザイン㈱</li>
              <li class="business__achievementItem">轟産業㈱</li>
              <li class="business__achievementItem">信越半導体㈱</li>
              <li class="business__achievementItem">エスエージャパンインコーポレイティッド</li>
              <li class="business__achievementItem">産業技術研究所</li>
            </ul>
          </div>
          <div class="business__wrap business__lineup">
            <h2 class="business__leading">製品ラインナップ</h2>
            <ul class="business__lineupList">
              <li class="business__lineupItem">
                <a href="">
                  <picture class="business__img">
                    <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/lineup/titan_sp.webp">
                    <img src="<?php echo get_template_directory_uri(); ?>/img/business/lineup/titan.webp" alt="">
                  </picture>
                </a>
              </li>
              <li class="business__lineupItem">
                <a href="">
                  <picture class="business__img">
                    <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/lineup/pure-water-high-precision-one-stiller_sp.webp">
                    <img src="<?php echo get_template_directory_uri(); ?>/img/business/lineup/pure-water-high-precision-one-stiller.webp" alt="">
                  </picture>
                </a>
              </li>
              <li class="business__lineupItem">
                <a href="">
                  <picture class="business__img">
                    <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/lineup/high-response_sp.webp">
                    <img src="<?php echo get_template_directory_uri(); ?>/img/business/lineup/high-response.webp" alt="">
                  </picture>
                </a>
              </li>
              <li class="business__lineupItem">
                <a href="">
                  <picture class="business__img">
                    <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/lineup/imagination-controll-tiller_sp.webp">
                    <img src="<?php echo get_template_directory_uri(); ?>/img/business/lineup/imagination-controll-tiller.webp" alt="">
                  </picture>
                </a>
              </li>
              <li class="business__lineupItem">
                <a href="">
                  <picture class="business__img">
                    <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/lineup/jacket-tank_sp.webp">
                    <img src="<?php echo get_template_directory_uri(); ?>/img/business/lineup/jacket-tank.webp" alt="">
                  </picture>
                </a>
              </li>
              <li class="business__lineupItem">
                <a href="">
                  <picture class="business__img">
                    <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/lineup/common-tiller_sp.webp">
                    <img src="<?php echo get_template_directory_uri(); ?>/img/business/lineup/common-tiller.webp" alt="">
                  </picture>
                </a>
              </li>
              <li class="business__lineupItem">
                <a href="">
                  <picture class="business__img">
                    <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/lineup/small-table-tiller_sp.webp">
                    <img src="<?php echo get_template_directory_uri(); ?>/img/business/lineup/small-table-tiller.webp" alt="">
                  </picture>
                </a>
              </li>
              <li class="business__lineupItem">
                <a href="">
                  <picture class="business__img">
                    <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/lineup/small-tiller_sp.webp">
                    <img src="<?php echo get_template_directory_uri(); ?>/img/business/lineup/small-tiller.webp" alt="">
                  </picture>
                </a>
              </li>
              <li class="business__lineupItem">
                <a href="">
                  <picture class="business__img">
                    <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/lineup/choice-tiller_sp.webp">
                    <img src="<?php echo get_template_directory_uri(); ?>/img/business/lineup/choice-tiller.webp" alt="">
                  </picture>
                </a>
              </li>
              <li class="business__lineupItem">
                <a href="">
                  <picture class="business__img">
                    <source media="(max-width: 870px)" srcset="<?php echo get_template_directory_uri(); ?>/img/business/lineup/separate_sp.webp">
                    <img src="<?php echo get_template_directory_uri(); ?>/img/business/lineup/separate.webp" alt="">
                  </picture>
                </a>
              </li>
            </ul>
          </div>
        </div>
      </section>
      <?php get_footer(); ?>