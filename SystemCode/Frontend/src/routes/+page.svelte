<script>
	import Submit from "$lib/Submit.svelte";

	const url = 'forest.webp';
</script>

<div class="w-full h-screen">
	<div class="hero w-full h-full" style="background-image: url({url});">
		<div class="hero-overlay bg-opacity-60" />
		<div class="hero-content w-full h-full flex-col justify-center items-center">
            <h1 class="mb-5 text-6xl font-bold text-white pb-8">🌲 Deforestation Detection</h1>
			<Submit />
		</div>
	</div>
</div>
