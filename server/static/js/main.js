/*
	Stellar by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
*/

(function($) {

	skel.breakpoints({
		xlarge: '(max-width: 1680px)',
		large: '(max-width: 1280px)',
		medium: '(max-width: 980px)',
		small: '(max-width: 736px)',
		xsmall: '(max-width: 480px)',
		xxsmall: '(max-width: 360px)'
	});

	$(function() {

		var	$window = $(window),
			$body = $('body'),
			$main = $('#main');

		// Disable animations/transitions until the page has loaded.
			$body.addClass('is-loading');

			$window.on('load', function() {
				window.setTimeout(function() {
					$body.removeClass('is-loading');
				}, 100);
			});

		// Fix: Placeholder polyfill.
			$('form').placeholder();

		// Prioritize "important" elements on medium.
			skel.on('+medium -medium', function() {
				$.prioritize(
					'.important\\28 medium\\29',
					skel.breakpoint('medium').active
				);
			});

		// Nav.
			var $nav = $('#nav');

			if ($nav.length > 0) {

				// Shrink effect.
					$main
						.scrollex({
							mode: 'top',
							enter: function() {
								$nav.addClass('alt');
							},
							leave: function() {
								$nav.removeClass('alt');
							},
						});

				// Links.
					var $nav_a = $nav.find('a');

					$nav_a
						.scrolly({
							speed: 1000,
							offset: function() { return $nav.height(); }
						})
						.on('click', function() {

							var $this = $(this);

							// External link? Bail.
								if ($this.attr('href').charAt(0) != '#')
									return;

							// Deactivate all links.
								$nav_a
									.removeClass('active')
									.removeClass('active-locked');

							// Activate link *and* lock it (so Scrollex doesn't try to activate other links as we're scrolling to this one's section).
								$this
									.addClass('active')
									.addClass('active-locked');

						})
						.each(function() {

							var	$this = $(this),
								id = $this.attr('href'),
								$section = $(id);

							// No section for this link? Bail.
								if ($section.length < 1)
									return;

							// Scrollex.
								$section.scrollex({
									mode: 'middle',
									initialize: function() {

										// Deactivate section.
											if (skel.canUse('transition'))
												$section.addClass('inactive');

									},
									enter: function() {

										// Activate section.
											$section.removeClass('inactive');

										// No locked links? Deactivate all links and activate this section's one.
											if ($nav_a.filter('.active-locked').length == 0) {

												$nav_a.removeClass('active');
												$this.addClass('active');

											}

										// Otherwise, if this section's link is the one that's locked, unlock it.
											else if ($this.hasClass('active-locked'))
												$this.removeClass('active-locked');

									}
								});

						});

			}

		// Scrolly.
			$('.scrolly').scrolly({
				speed: 1000
			});

	});

lst = [['Cardiomegaly/mild', 'Aorta/tortos', 'Opacity/right/paratracheal', 'Osteophyte/thoracic vertebrae/degenerative', 'Spondylosis/thoracic vertebrae'],
['normal'],
['Aorta/tortos', 'Sarcoidosis', 'Cicatrix/lng/apex/bilateral'],
['Fractres, Bone/ribs/right/healed', 'Density/bilateral/rond/small'],
['Lng/hypoinflation'],
['Opacity/lng/base/bilateral', 'Plmonary Atelectasis/base/bilateral'],
['Calcified Granloma/scattered/mltiple', 'Implanted Medical Device/hmers/right'],
['Calcinosis/aorta', 'Granloma/lng/lingla', 'Spondylosis/thoracic vertebrae', 'Nodle/lng/lingla'],
['Opacity/lng/lingla', 'Deformity/thoracic vertebrae/mild'],
['Diaphragm/right/elevated', 'Consolidation/lng/base/right', 'Pulmonary Atelectasis/base/right', 'Airspace Disease/lung/lower lobe/right'],
['Granuloma/ribs/right/posterior','Opacity/ribs/right/posterior/round'],
['Granulomatous Disease']];

	$("input").click(
    function() {

			if($("#new").length == 0) {
        setTimeout(
            function() {

								$('#cta').append('<div id="new" style="margin-top:30px;height:30px;font-size:30px;">'+lst[Math.floor(Math.random() * 11) + 0] +'</div>');
							},5000
						)}
			else if($("#new").length == 1  ){
					document.getElementById("new").remove();
					setTimeout(
							function() {

									$('#cta').append('<div id="new" style="margin-top:30px;height:30px;font-size:30px;">'+lst[Math.floor(Math.random() * 11) + 0] +'</div>');
								},5000
							)
				}

            });

})(jQuery);
