<script>
    /**
     * @type {string[]}
     */
    export let data;
    /**
     * @type {string}
     */
    export let name;
    /**
     * @type {string}
     */
    export let date;

    let selected = data.default;
    let images = data.images;
    const labelMap = [
        "Cloud", "Forest", "Grassland",
        "Wetland", "Urban",
        "Barren", "Water", "Origin"
    ]
</script>

<div class="w-full h-fit bg-base-100">
    <div class="flex flex-row items-center">
        <h1 class="text-3xl flex-1 font-bold">{name}</h1>
        <h3 class="text-xl pr-8">Date:</h3>
        <div class="badge badge-secondary badge-outline badge-lg">{date}</div>
    </div>
    <div class="divider"></div>
    <div class="items-center ">
        <div class="flex justify-center w-full">
            {#each images as image, id}
                {#if id === selected}
                    <img src={ images[id] ? `data:image/png;base64,${image}` : ""} alt="missing"/>
                {/if}
            {/each}
        </div>
        
        <div class="flex justify-center w-full py-4 space-x-8">
            <div class="join">
                <label>
                    {#each images as _, id}
                        <input 
                            bind:group={selected} 
                            value={id} 
                            class={ images[id] ? "join-item btn normal-case" : "join-item btn btn-disabled normal-case" } type="radio" name="options" 
                            aria-label={labelMap[id]}
                        />
                    {/each}
                </label>
            </div>
        </div>
    </div>
</div>