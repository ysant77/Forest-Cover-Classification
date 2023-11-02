<script>
	import Image from "$lib/Image.svelte";

    export let data;

    const status = data.status;
    const places = data.places;
    const dates = data.dates;
</script>

<div class="navbar h-8 bg-primary-focus text-primary-content flex w-full">
    <div class="flex-1 pl-8">
        <a class="btn btn-ghost normal-case text-xl text-white" href="/">🌲 Deforestation Detection</a>
    </div>
    <div class="flex-none pr-8">
    </div>
</div>

<div class="w-full min-h-screen bg-base-100 flex flex-row">
    <div class="w-[55%] flex-col bg-base-100 p-8">
        {#if status === 200}
            <Image data={data.first} name={places[0]} date={dates[0]}/>
            <Image data={data.second} name={places[1]} date={dates[1]}/>
        {/if}

        {#if status === 400}
            <h2>{data.message}</h2>
        {/if}
    </div>
    <div class="w-[45%] bg-base-100 p-8 border-l-4">
        <h1 class="text-2xl">Quantify Changes</h1>
        <div class="divider"></div>
        <div class="overflow-x-auto">
            <table class="table">
              <!-- head -->
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Change in Pixels</th>
                  <th>Change in Area (sq km)</th>
                  <th>Change in Percentage</th>
                </tr>
              </thead>
              <tbody>
                <!-- row 1 -->
                {#each data.changes as change}
                    <tr class="hover">
                        <th>{change.Class}</th>
                        <td style="color: {change.change_in_pixels < 0 ? 'red' : change.change_in_pixels > 0 ? 'green' : 'black'}">{change.change_in_pixels}</td>
                        <td style="color: {change.change_in_area < 0 ? 'red' : change.change_in_area > 0 ? 'green' : 'black'}">{change.change_in_area}</td>
                        <td style="color: {change.change_in_percentage < 0 ? 'red' : change.change_in_percentage > 0 ? 'green' : 'black'}">{change.change_in_percentage.toFixed(3)}</td>
                    </tr>
                {/each}
              </tbody>
            </table>
        </div>
        <h1 class="text-2xl pt-8">Export Results</h1>
        <div class="divider"></div>
        <div class="grid grid-flow-col space-x-8">
            <a class="btn btn-accent normal-case text-white" href="http://localhost:8000/download/report">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                </svg>              
                Report
            </a>
            <a class="btn btn-accent normal-case text-white" href="http://localhost:8000/download/analysis">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                </svg>              
                Images
            </a>
        </div>
    </div>
</div>