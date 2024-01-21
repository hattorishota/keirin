<?php

// mw wp form のpタグ自動挿入を防ぐ
function mvwpform_autop_filter() {
    if (class_exists('MW_WP_Form_Admin')) {
      $mw_wp_form_admin = new MW_WP_Form_Admin();
      $forms = $mw_wp_form_admin->get_forms();
      foreach ($forms as $form) {
        add_filter('mwform_content_wpautop_mw-wp-form-' . $form->ID, '__return_false');
      }
    }
  }
mvwpform_autop_filter();

// 新着記事にNEWを付加する
// NEWを表示する日数を指定
define("NEWEST_POST_DAYS", 7);
  
// 記事にNEWを付けるか判定する処理
function is_newest_post($the_post)
{
    // NEWを付加する日数
    $days = NEWEST_POST_DAYS;

    // 記事投稿後の経過日数
    $today = date_i18n('U');
    $posted = get_the_time('U', $the_post->ID);
    $elapsed = date('U', ($today - $posted)) / (60 * 60 * 24);

    // NEWを付加する日数よりも経過日が小さければtrueを返す
    if ($days > $elapsed) {
        return true;
    } else {
        return false;
    }
}

// 管理画面から投稿を非表示にする
function remove_menus()
{
	global $menu;
	remove_menu_page('edit.php');
}
add_action('admin_menu', 'remove_menus');

// アイキャッチ画像設定
function my_theme_setup() {
    add_theme_support('post-thumbnails');
  }
  add_action( 'after_setup_theme', 'my_theme_setup');
  
  // 記事文抜粋の文字数を変更
  function custom_excerpt_length( $length ) {
      return 36;
  }
  function new_excerpt_more( $more ) {
      return '...' ;
  }
add_filter( 'excerpt_length' , 'custom_excerpt_length' , 999 );
add_filter( 'excerpt_more' , 'new_excerpt_more' );

// カスタム投稿タイプの追加
function custum_post_type()
{
	// お知らせ
	register_post_type(
		'news_post',
		array(
			'labels' =>
			array(
				'name' => __('お知らせ'),
				'singular_name' => __('news_post')
			),
			'public' => true,
			'menu_position' => 4,
			'has_archive' => true,
			'supports' => array('title','editor','thumbnail'),
			'show_in_rest' => true,
		)
	);
}
add_action('init', 'custum_post_type');

// カスタムタクソノミーの追加
function create_category_taxonomy()
{
	// お知らせ用カテゴリー
	$args = array(
		'label' => 'カテゴリー',
		'public' => true,
		'hierarchical' => true,
		'show_in_rest' => true,
		'show_admin_column' => true,
	);
	register_taxonomy('news_cat', 'news_post', $args);

	// お知らせ用タグ
	$args = array(
		'label' => 'タグ',
		'public' => true,
		'hierarchical' => true,
		'show_in_rest' => true,
		'show_admin_column' => true,
	);
	register_taxonomy('news_tag', 'news_post', $args);
}
add_action('init', 'create_category_taxonomy');
