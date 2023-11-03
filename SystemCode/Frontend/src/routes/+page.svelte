<script>
	import Submit from "$lib/Submit.svelte";
	export let form;

	const url = 'forest.webp';
</script>

<div class="w-full h-screen">
	<div class="hero w-full h-full" style="background-image: url({url});">
		<div class="hero-overlay bg-opacity-60" />
		<div class="hero-content w-full h-full flex-col justify-center items-center">
			{#if form?.error}
				<div class="alert alert-error">
					<svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
					<span>Error! {form.error}</span>
				</div>
			{/if}
            <h1 class="mb-5 text-6xl font-bold text-white pb-8">🌲 Deforestation Detection</h1>
			<Submit />
		</div>
	</div>
</div>
